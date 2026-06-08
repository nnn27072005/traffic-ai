import time
import cv2
import numpy as np
import subprocess
import os
import asyncio
import sys
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

from src.api.pipeline import TrafficPipeline

# Config
VIDEO_PATH = "dummy_traffic.mp4"
TRAFFIC_MODEL = os.getenv("TRAFFIC_MODEL", "runs/train/fisheye8k-yolov10s-1280/weights/best.pt")
HELMET_MODEL = os.getenv("HELMET_MODEL", "runs/train/helmet-yolov10s-640/weights/best.pt")
NUM_FRAMES = 50 # Lower frames for quicker benchmark

async def benchmark():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file {VIDEO_PATH} not found.")
        return

    # Probe video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"=== Traffic-AI Pipeline Benchmark ===")
    print(f"Video: {VIDEO_PATH} ({width}x{height} @ {fps_orig} FPS)")
    print(f"Models: {TRAFFIC_MODEL} / {HELMET_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"======================================")
    
    # CASE 1: Ingestion Speed
    print("\n[CASE 1] Ingestion Speed (Decoding Only)")
    
    # OpenCV
    cap = cv2.VideoCapture(VIDEO_PATH)
    t0 = time.perf_counter()
    count = 0
    while count < NUM_FRAMES:
        ret, _ = cap.read()
        if not ret: break
        count += 1
    t1 = time.perf_counter()
    cap.release()
    print(f"  - OpenCV Decode: {count/(t1-t0):.2f} FPS")
    
    # FFmpeg Pipe
    ffmpeg_available = False
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffmpeg_available = True
    except FileNotFoundError:
        print("  - FFmpeg Pipe: NOT FOUND (skipping)")

    if ffmpeg_available:
        ingest_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', VIDEO_PATH, 
            '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ]
        t0 = time.perf_counter()
        proc = subprocess.Popen(ingest_cmd, stdout=subprocess.PIPE, bufsize=width*height*3*10)
        count = 0
        frame_size = width * height * 3
        while count < NUM_FRAMES:
            raw = proc.stdout.read(frame_size)
            if not raw or len(raw) != frame_size: break
            count += 1
        t1 = time.perf_counter()
        proc.terminate()
        print(f"  - FFmpeg Pipe Decode: {count/(t1-t0):.2f} FPS")

    # Load Pipeline
    print("\nLoading Models...")
    try:
        pipeline = TrafficPipeline(TRAFFIC_MODEL, HELMET_MODEL, device=DEVICE)
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return
    
    # CASE 2: Decode + YOLO
    print("\n[CASE 2] Decode + YOLO Detection")
    cap = cv2.VideoCapture(VIDEO_PATH)
    t0 = time.perf_counter()
    count = 0
    while count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        _ = pipeline.detector.detect(frame)
        count += 1
    t1 = time.perf_counter()
    cap.release()
    print(f"  - Speed: {count/(t1-t0):.2f} FPS")

    # CASE 3: Decode + YOLO + ByteTrack
    print("\n[CASE 3] Decode + YOLO + ByteTrack")
    pipeline.reset()
    cap = cv2.VideoCapture(VIDEO_PATH)
    t0 = time.perf_counter()
    count = 0
    while count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        detections = pipeline.detector.detect(frame)
        _ = pipeline.tracker.update(detections, frame.shape[:2])
        count += 1
    t1 = time.perf_counter()
    cap.release()
    print(f"  - Speed: {count/(t1-t0):.2f} FPS")

    # CASE 4: + Helmet (Synchronous - Legacy)
    print("\n[CASE 4] Decode + YOLO + Track + Sync Helmet Analysis (Legacy)")
    pipeline.reset()
    pipeline.violation_queue = None # Force sync
    cap = cv2.VideoCapture(VIDEO_PATH)
    t0 = time.perf_counter()
    count = 0
    while count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        result = pipeline.process_frame(frame)
        count += 1
    t1 = time.perf_counter()
    cap.release()
    print(f"  - Speed: {count/(t1-t0):.2f} FPS")

    # CASE 5: Optimized Full Pipeline
    print("\n[CASE 5] Full Optimized Pipeline (Async Violation + Preview Render)")
    pipeline.reset()
    pipeline.violation_queue = asyncio.Queue()
    
    # Mock worker that just drains (simulation)
    async def mock_worker():
        while True:
            item = await pipeline.violation_queue.get()
            if item is None: break
            # Simulate secondary AI workload without blocking main loop
            _ = pipeline.analyzer.analyze(item["frame"], item["tracks"], item["frame_idx"])
            pipeline.violation_queue.task_done()

    worker_task = asyncio.create_task(mock_worker())
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    t0 = time.perf_counter()
    count = 0
    while count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        # Process frame (Primary AI)
        result = pipeline.process_frame(frame)
        # Lightweight Render
        _ = pipeline.render(frame, result, mode="preview")
        count += 1
    t1 = time.perf_counter()
    cap.release()
    
    await pipeline.violation_queue.put(None)
    await worker_task
    print(f"  - Speed: {count/(t1-t0):.2f} FPS")

    # CASE 6: Full Pipeline + Export
    print("\n[CASE 6] Full Optimized Pipeline + Full-Res Export Rendering")
    pipeline.reset()
    pipeline.violation_queue = asyncio.Queue()
    worker_task = asyncio.create_task(mock_worker())
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    t0 = time.perf_counter()
    count = 0
    while count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        result = pipeline.process_frame(frame)
        _ = pipeline.render(frame, result, mode="preview")
        _ = pipeline.render(frame, result, mode="export")
        count += 1
    t1 = time.perf_counter()
    cap.release()
    await pipeline.violation_queue.put(None)
    await worker_task
    print(f"  - Speed: {count/(t1-t0):.2f} FPS")

    print("\nBenchmark Complete.")

if __name__ == "__main__":
    asyncio.run(benchmark())
