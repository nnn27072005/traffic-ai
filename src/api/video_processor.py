# src/api/video_processor.py
import asyncio
import cv2
import logging
import time
import subprocess
import numpy as np
import os
from sqlalchemy.orm import Session
from src.api import crud, database
from src.api.pipeline import TrafficPipeline
from src.api.stream_manager import stream_manager
from src.api.websocket_manager import websocket_manager

logger = logging.getLogger("traffic-api.video_processor")

async def violation_worker(
    pipeline: TrafficPipeline,
    session_id: int,
    save_evidence: bool = False,
):
    """
    Background worker to process helmet violations asynchronously.
    Prevents secondary AI tasks from blocking the main traffic pipeline.
    """
    logger.info(f"Violation worker started for session {session_id}")
    try:
        while True:
            item = await pipeline.violation_queue.get()
            if item is None: # Poison pill
                pipeline.violation_queue.task_done()
                break
            
            frame = item["frame"]
            tracks = item["tracks"]
            frame_idx = item["frame_idx"]
            should_save_evidence = bool(item.get("save_evidence", save_evidence))
            
            # Analyze violations (Heavy AI call)
            # This runs in the worker thread/task, not blocking ingestion
            violations = pipeline.analyzer.analyze(frame, tracks, frame_idx)
            
            if violations:
                class_name_by_track = {
                    track.track_id: track.class_name
                    for track in tracks
                }
                with database.SessionLocal() as db:
                    for v in violations:
                        # Convert dataclass to dict and format bbox for crud
                        v_data = {
                            "track_id": v.track_id,
                            "violations": v.violations,
                            "confidence": v.confidence,
                            "bbox": {
                                "x1": float(v.bbox_xyxy[0]), 
                                "y1": float(v.bbox_xyxy[1]), 
                                "x2": float(v.bbox_xyxy[2]), 
                                "y2": float(v.bbox_xyxy[3])
                            },
                            "class_name": class_name_by_track.get(v.track_id, "Bike"),
                            "frame_id": frame_idx
                        }
                        db_violation = crud.create_violation(
                            db,
                            v_data,
                            session_id=session_id,
                            frame=frame if should_save_evidence else None
                        )
                        await websocket_manager.broadcast(session_id, {
                            "type": "violation",
                            "session_id": session_id,
                            "frame_id": frame_idx,
                            "track_id": v.track_id,
                            "violations": v.violations,
                            "confidence": round(float(v.confidence), 3),
                            "bbox": v_data["bbox"],
                            "class_name": v_data["class_name"],
                            "violation_id": db_violation.id,
                            "asset_url": db_violation.asset_url,
                            "save_evidence": should_save_evidence,
                        })
            pipeline.violation_queue.task_done()
    except Exception as e:
        logger.error(f"Error in violation worker for session {session_id}: {e}")
    finally:
        logger.info(f"Violation worker stopped for session {session_id}")

async def process_video_file(
    video_path: str,
    session_id: int,
    pipeline: TrafficPipeline,
    line_ratio: float = 0.5,
    frame_skip: int = 2,
    target_analysis_fps: float | None = None,
    enable_export: bool = False,
    save_evidence: bool = False,
):
    """
    High-performance video processing using FFmpeg pipes and async workers.
    """
    from src.api.database import ASSETS_DIR

    logger.info(f"Starting optimized video processing for session {session_id}")
    await websocket_manager.broadcast(session_id, {
        "type": "session_started",
        "session_id": session_id,
        "save_output": enable_export,
        "save_evidence": save_evidence,
    })
    
    # 1. Probe video metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to probe video file: {video_path}")
        with database.SessionLocal() as db:
            crud.update_session(db, session_id, status="failed", error_message="Failed to open video file")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fps = fps or 25.0

    if target_analysis_fps is None:
        target_analysis_fps = float(os.getenv("VIDEO_ANALYSIS_FPS", "5"))
    effective_frame_skip = max(1, frame_skip, int(round(fps / max(target_analysis_fps, 0.1))))
    output_fps = max(fps / effective_frame_skip, 1.0)
    logger.info(
        "Session %s video metadata: %sx%s @ %.2f FPS, %s frames. "
        "Analyzing every %s frame(s) (~%.2f FPS).",
        session_id,
        width,
        height,
        fps,
        total_frames,
        effective_frame_skip,
        output_fps,
    )

    # 2. Setup Ingestion (FFmpeg Pipe or OpenCV Fallback)
    ffmpeg_available = False
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffmpeg_available = True
    except FileNotFoundError:
        logger.warning("FFmpeg not found in PATH. Falling back to OpenCV VideoCapture.")

    ingest_proc = None
    cap = None
    if ffmpeg_available:
        ingest_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', video_path,
            '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ]
        ingest_proc = subprocess.Popen(ingest_cmd, stdout=subprocess.PIPE, bufsize=width * height * 3 * 10)
    else:
        cap = cv2.VideoCapture(video_path)

    # 3. Setup Export (FFmpeg Pipe or OpenCV Fallback)
    out_proc = None
    out_writer = None
    output_rel_path = None
    if enable_export:
        output_dir = ASSETS_DIR / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"session_{session_id}.mp4"
        output_rel_path = f"outputs/{output_filename}"
        output_abs_path = str(output_dir / output_filename)
        
        if ffmpeg_available:
            # Try NVENC if available, fallback to libx264
            encoder = 'h264_nvenc' 
            export_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{width}x{height}", '-r', f"{output_fps}",
                '-i', '-',
                '-c:v', encoder, '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                output_abs_path
            ]
            try:
                out_proc = subprocess.Popen(export_cmd, stdin=subprocess.PIPE, bufsize=width * height * 3 * 10)
            except Exception:
                logger.warning("NVENC not found, falling back to libx264")
                export_cmd[16] = 'libx264'
                out_proc = subprocess.Popen(export_cmd, stdin=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_abs_path, fourcc, output_fps, (width, height))

    # 4. Initialize Workers
    pipeline.reset()
    pipeline.violation_queue = asyncio.Queue(maxsize=100)
    violation_task = asyncio.create_task(
        violation_worker(pipeline, session_id, save_evidence=save_evidence)
    )

    with database.SessionLocal() as db:
        crud.update_session(db, session_id, status="processing", total_frames=total_frames, fps=fps)
    await websocket_manager.broadcast(session_id, {
        "type": "session_status",
        "session_id": session_id,
        "status": "processing",
        "total_frames": total_frames,
        "processed_frames": 0,
        "fps": fps,
        "analysis_fps": target_analysis_fps,
    })

    source_frame_count = 0
    analyzed_count = 0
    frame_size = width * height * 3
    last_result = None
    failed = False
    terminal_status = "completed"
    
    try:
        while True:
            # Read frame (FFmpeg or OpenCV)
            if ingest_proc:
                raw_frame = ingest_proc.stdout.read(frame_size)
                if not raw_frame or len(raw_frame) != frame_size:
                    break
                frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            source_frame_count += 1

            if source_frame_count % effective_frame_skip != 0:
                if source_frame_count % (effective_frame_skip * 20) == 0:
                    with database.SessionLocal() as db:
                        current_session = crud.get_session(db, session_id)
                        if current_session and current_session.status in ("stopped", "cancelled"):
                            terminal_status = current_session.status
                            break
                        crud.update_session(db, session_id, processed_frames=source_frame_count)
                continue
            
            # Process frame (Primary Pipeline Only)
            # This is now much faster because violation analysis is offloaded
            result = pipeline.process_frame(
                frame, 
                line_ratio=line_ratio, 
                session_id=session_id,
                save_evidence=save_evidence,
            )
            last_result = result
            analyzed_count += 1
            
            # 5. Export path (Full-res). Reuse the same rendered frame for
            # preview instead of running the annotator twice.
            if enable_export:
                export_frame = pipeline.render(frame, result, mode="export", line_ratio=line_ratio)
                if out_proc and out_proc.stdin:
                    out_proc.stdin.write(export_frame.tobytes())
                elif out_writer:
                    out_writer.write(export_frame)
                preview_frame = export_frame
                if width > 1280:
                    preview_h = int(height * (1280 / width))
                    preview_frame = cv2.resize(export_frame, (1280, preview_h))
            else:
                preview_frame = pipeline.render(frame, result, mode="preview", line_ratio=line_ratio)
            stream_manager.update_frame(session_id, preview_frame)
            
            # periodic DB updates
            if analyzed_count % 20 == 0:
                with database.SessionLocal() as db:
                    current_session = crud.get_session(db, session_id)
                    if current_session and current_session.status in ("stopped", "cancelled"):
                        terminal_status = current_session.status
                        break
                    
                    crud.create_traffic_stats(db, result.model_dump(), session_id=session_id)
                    crud.update_session(db, session_id, processed_frames=source_frame_count)
                await websocket_manager.broadcast(session_id, {
                    "type": "progress",
                    "session_id": session_id,
                    "status": "processing",
                    "total_frames": total_frames,
                    "processed_frames": source_frame_count,
                    "analyzed_frames": analyzed_count,
                    "latency_ms": result.latency_ms,
                    "vehicle_count": result.vehicle_count,
                    "count_in": result.count_in,
                    "count_out": result.count_out,
                })
            
            # Yield to event loop
            if analyzed_count % 10 == 0:
                await asyncio.sleep(0.001)

        if last_result is not None:
            with database.SessionLocal() as db:
                crud.create_traffic_stats(db, last_result.model_dump(), session_id=session_id)

        logger.info(
            "Processing loop finished for session %s: read %s source frames, analyzed %s frames",
            session_id,
            source_frame_count,
            analyzed_count,
        )
        
    except Exception as e:
        failed = True
        logger.exception(f"Error processing video session {session_id}")
        with database.SessionLocal() as db:
            crud.update_session(db, session_id, status="failed", error_message=str(e))
        await websocket_manager.broadcast(session_id, {
            "type": "session_status",
            "session_id": session_id,
            "status": "failed",
            "error_message": str(e),
        })
    finally:
        # Cleanup
        if ingest_proc:
            ingest_proc.terminate()
        if cap:
            cap.release()
            
        if out_proc:
            if out_proc.stdin:
                out_proc.stdin.close()
            out_proc.wait()
        if out_writer:
            out_writer.release()
        
        # Stop violation worker
        await pipeline.violation_queue.put(None)
        await violation_task
        
        if not failed:
            with database.SessionLocal() as db:
                crud.update_session(
                    db, session_id, 
                    status=terminal_status,
                    processed_frames=source_frame_count,
                    output_video_path=output_rel_path
                )
            await websocket_manager.broadcast(session_id, {
                "type": "session_status",
                "session_id": session_id,
                "status": terminal_status,
                "total_frames": total_frames,
                "processed_frames": source_frame_count,
                "analyzed_frames": analyzed_count,
                "output_video_path": output_rel_path,
            })
        
        stream_manager.cleanup_session(session_id)

import asyncio # Needed for await
