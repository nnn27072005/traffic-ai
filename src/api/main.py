# src/api/main.py
from __future__ import annotations
import os, io, time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from src.api.pipeline import TrafficPipeline
from src.api.schemas  import AnalysisResult, HealthResponse, BenchmarkResponse, BenchmarkRow


# ── Global pipeline instance ──────────────────────────────────────
pipeline: TrafficPipeline | None = None

TRAFFIC_MODEL = os.getenv(
    "TRAFFIC_MODEL",
    "/kaggle/working/traffic-ai/runs/train/fisheye8k-yolov10s-1280/weights/best.pt",
)
HELMET_MODEL = os.getenv(
    "HELMET_MODEL",
    "/kaggle/working/traffic-ai/runs/train/helmet-yolov10s-640/weights/best.pt",
)
DEVICE = os.getenv("DEVICE", "cuda")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global pipeline
    pipeline = TrafficPipeline(
        traffic_model_path = TRAFFIC_MODEL,
        helmet_model_path  = HELMET_MODEL,
        device             = DEVICE,
    )
    yield
    # Cleanup
    pipeline = None


app = FastAPI(
    title       = "Traffic Safety Monitoring API",
    description = "End-to-end traffic analytics: detection, tracking, helmet violation",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Kiểm tra API status và models đã load chưa."""
    return HealthResponse(
        status        = "ok" if pipeline is not None else "loading",
        traffic_model = TRAFFIC_MODEL,
        helmet_model  = HELMET_MODEL,
        device        = DEVICE,
    )


@app.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image(
    file:       UploadFile = File(...),
    line_ratio: float      = Query(default=0.5, ge=0.1, le=0.9),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check sau khi decode — nếu không đọc được thì mới báo lỗi
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    result = pipeline.process_frame(frame, line_ratio=line_ratio)
    return result


@app.post("/analyze/image/annotated")
async def analyze_image_annotated(
    file: UploadFile = File(...),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    result = pipeline.process_frame(frame)

    from src.analyzer.violation import draw_violations
    import supervision as sv

    if result.tracks:
        bboxes = np.array([[t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2]
                           for t in result.tracks])
        sv_det  = sv.Detections(xyxy=bboxes)
        box_ann = sv.BoxAnnotator(thickness=2)
        frame   = box_ann.annotate(frame, sv_det)

    for v in result.violations:
        x1, y1 = int(v.bbox.x1), int(v.bbox.y1)
        x2, y2 = int(v.bbox.x2), int(v.bbox.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"VIOLATION {v.confidence:.0%}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/jpeg",
    )


@app.post("/reset")
async def reset_pipeline():
    """Reset tracker và counter — dùng khi chuyển video source."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    pipeline.reset()
    return {"status": "reset", "message": "Tracker and counter cleared"}


@app.get("/benchmark", response_model=BenchmarkResponse)
async def get_benchmark():
    """Trả về benchmark results đã chạy ở Phase 5."""
    import json
    bench_path = "/kaggle/working/traffic-ai/runs/export/benchmark_results.json"
    try:
        data = json.loads(open(bench_path).read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark results not found")

    rows = [
        BenchmarkRow(name="Traffic — PyTorch GPU",
                     mean_ms=data["traffic_model"]["pytorch_gpu_ms"],
                     fps=data["traffic_model"]["pytorch_gpu_fps"]),
        BenchmarkRow(name="Traffic — ONNX CPU",
                     mean_ms=data["traffic_model"]["onnx_cpu_ms"],
                     fps=data["traffic_model"]["onnx_cpu_fps"]),
        BenchmarkRow(name="Helmet — PyTorch GPU",
                     mean_ms=data["helmet_model"]["pytorch_gpu_ms"],
                     fps=data["helmet_model"]["pytorch_gpu_fps"]),
        BenchmarkRow(name="Helmet — ONNX CPU",
                     mean_ms=data["helmet_model"]["onnx_cpu_ms"],
                     fps=data["helmet_model"]["onnx_cpu_fps"]),
    ]
    return BenchmarkResponse(gpu=data["gpu"], results=rows)