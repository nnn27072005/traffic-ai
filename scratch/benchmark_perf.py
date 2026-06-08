import time
import cv2
import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.pipeline import TrafficPipeline

def benchmark():
    device = "cpu"
    traffic_model = "runs/train/fisheye8k-yolov10s-1280/weights/best.pt"
    helmet_model = "runs/train/helmet-yolov10s-640/weights/best.pt"
    
    print(f"Initializing pipeline on {device}...")
    pipeline = TrafficPipeline(
        traffic_model_path=traffic_model,
        helmet_model_path=helmet_model,
        device=device
    )
    
    # Create a dummy frame
    frame = (torch.randn(720, 1280, 3) * 255).numpy().astype("uint8")
    
    print("Starting benchmark (10 frames)...")
    start_time = time.time()
    for i in range(10):
        t1 = time.time()
        pipeline.process_frame(frame, annotate=True)
        t2 = time.time()
        print(f"Frame {i+1}: {(t2-t1)*1000:.2f}ms")
    
    end_time = time.time()
    avg_latency = (end_time - start_time) / 10
    print(f"\nAverage Latency: {avg_latency*1000:.2f}ms")
    print(f"Estimated Throughput: {1/avg_latency:.2f} FPS")
    
    video_duration_sec = 137
    fps_video = 30
    total_frames = video_duration_sec * fps_video
    
    for skip in [2, 5, 10]:
        proc_frames = total_frames / skip
        total_time_min = (proc_frames * avg_latency) / 60
        print(f"With Frame Skip {skip}: Estimated total time = {total_time_min:.1f} minutes")

if __name__ == "__main__":
    benchmark()
