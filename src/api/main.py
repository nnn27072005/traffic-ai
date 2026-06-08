# src/api/main.py
from __future__ import annotations
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.pipeline import TrafficPipeline
from src.api.schemas  import AnalysisResult, HealthResponse, BenchmarkResponse, BenchmarkRow
from src.api import crud, database, models
from src.api.database import engine, get_db
from src.api.routers import cameras, violations, sessions, analytics, stream, auth, video, ws
from src.api.stream_manager import stream_manager
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np

# Database tables are managed via Alembic migrations.
# models.Base.metadata.create_all(bind=engine)

# ── Logging Setup ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traffic-api")

# ── Global pipeline instance ──────────────────────────────────────
pipeline: TrafficPipeline | None = None

TRAFFIC_MODEL = os.getenv("TRAFFIC_MODEL", "runs/train/fisheye8k-yolov10s-1280/weights/best.pt")
HELMET_MODEL = os.getenv("HELMET_MODEL", "runs/train/helmet-yolov10s-640/weights/best.pt")
DEVICE = os.getenv("DEVICE", "cuda")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global pipeline
    logger.info(f"Loading models to {DEVICE}...")
    try:
        pipeline = TrafficPipeline(
            traffic_model_path = TRAFFIC_MODEL,
            helmet_model_path  = HELMET_MODEL,
            device             = DEVICE,
        )
        logger.info("Pipeline ready.")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
    
    yield
    # Cleanup
    pipeline = None

app = FastAPI(
    title       = "Traffic Safety Monitoring API",
    description = "Modularized Traffic Analytics: detection, tracking, helmet violation",
    version     = "2.0.0",
    lifespan    = lifespan,
)

# ── CORS Middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Include Routers ──────────────────────────────────────────────
app.include_router(cameras.router, prefix="/api")
app.include_router(violations.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(stream.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(video.router, prefix="/api")
app.include_router(ws.router, prefix="/api")

# ── Static Files ────────────────────────────────────────────────
app.mount("/api/assets", StaticFiles(directory="data/assets"), name="assets")

# ── Legacy/Root Endpoints ────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status        = "ok" if pipeline is not None else "loading",
        traffic_model = TRAFFIC_MODEL,
        helmet_model  = HELMET_MODEL,
        device        = DEVICE,
    )

@app.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    line_ratio: float = Query(default=0.5, ge=0.1, le=0.9),
    session_id: int = Query(default=1), # Default to session 1 for testing
    db: Session = Depends(get_db)
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    # For one-off images, we use a default camera or create one
    camera = crud.get_or_create_default_camera(db)
    
    # Process frame
    result = pipeline.process_frame(
        frame, 
        line_ratio=line_ratio, 
        session_id=session_id, 
        camera_id=camera.id
    )

    # Render preview
    annotated_frame = pipeline.render(frame, result, mode="preview", line_ratio=line_ratio)

    if annotated_frame is not None:
        stream_manager.update_frame(session_id, annotated_frame)
    
    # Save statistics
    crud.create_traffic_stats(db, {
        "vehicle_count": result.vehicle_count,
        "count_in": result.count_in,
        "count_out": result.count_out,
        "latency_ms": result.latency_ms,
        "frame_id": result.frame_id
    }, camera_id=camera.id)
    
    # Save violations (now with auto-crop saving!)
    for v in result.violations:
        violation_data = {
            "track_id": v.track_id,
            "violations": v.violations,
            "confidence": v.confidence,
            "bbox": {"x1": v.bbox.x1, "y1": v.bbox.y1, "x2": v.bbox.x2, "y2": v.bbox.y2},
            "class_name": next((t.class_name for t in result.tracks if t.track_id == v.track_id), "unknown"),
            "frame_id": result.frame_id
        }
        crud.create_violation(db, violation_data, camera_id=camera.id, frame=frame)

    return result

@app.post("/reset")
async def reset_pipeline():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    pipeline.reset()
    return {"status": "reset", "message": "Tracker and counter cleared"}

@app.get("/benchmark", response_model=BenchmarkResponse)
async def get_benchmark():
    import json
    bench_path = os.getenv("BENCHMARK_PATH", "runs/export/benchmark_results.json")
    if not os.path.exists(bench_path):
        raise HTTPException(status_code=404, detail="Benchmark results not found")
    with open(bench_path) as f:
        data = json.load(f)
    
    rows = []
    # Simplified mapping for demonstration
    for key, val in data.items():
        if isinstance(val, dict) and "pytorch_gpu_fps" in val:
            rows.append(BenchmarkRow(name=f"{key} (GPU)", mean_ms=val.get("pytorch_gpu_ms", 0), fps=val["pytorch_gpu_fps"]))
    
    return BenchmarkResponse(gpu=data.get("gpu", "unknown"), results=rows)

# Need an empty __init__.py in routers
