import asyncio
import os
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.pipeline import TrafficPipeline
from src.api.utils.video import get_stream_url
from src.api.video_processor import process_video_file
from src.api.database import SessionLocal, Base, engine
from src.api import models

# Config
YOUTUBE_URL = "https://www.youtube.com/watch?v=fm0cX7P6PSw"
TRAFFIC_MODEL = os.getenv("TRAFFIC_MODEL", "runs/train/fisheye8k-yolov10s-1280/weights/best.pt")
HELMET_MODEL = os.getenv("HELMET_MODEL", "runs/train/helmet-yolov10s-640/weights/best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

async def test_youtube():
    print(f"=== Testing YouTube Processing Pipeline ===")
    print(f"URL: {YOUTUBE_URL}")
    print(f"Device: {DEVICE}")

    # 1. Extract Stream URL
    print("\n[Step 1] Extracting Stream URL...")
    try:
        stream_url = get_stream_url(YOUTUBE_URL)
        print(f"  Success! Stream URL extracted.")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # 2. Initialize Pipeline
    print("\n[Step 2] Initializing Pipeline...")
    pipeline = TrafficPipeline(TRAFFIC_MODEL, HELMET_MODEL, device=DEVICE)
    print("  Pipeline Ready.")

    # 3. Create a dummy session in DB
    print("\n[Step 3] Creating Test Session...")
    Base.metadata.create_all(bind=engine)
    from src.api import crud
    with SessionLocal() as db:
        session = crud.create_session(
            db, 
            source_type="url", 
            source_path=stream_url,
            config={"original_url": YOUTUBE_URL}
        )
        session_id = session.id
        print(f"  Session Created: ID={session_id}")

    # 4. Process Video (Run for limited frames to test)
    print("\n[Step 4] Starting Video Processor...")
    print("  (Processing will run for a few seconds to verify functionality)")
    
    # We'll use a modified version of process_video_file or just run it and terminate
    try:
        # Running the optimized process_video_file
        # This will use FFmpeg pipe (if available) or OpenCV fallback
        task = asyncio.create_task(process_video_file(
            video_path=stream_url,
            session_id=session_id,
            pipeline=pipeline,
            frame_skip=5, # Skip frames for speed
            enable_export=True
        ))
        
        # Wait for 15 seconds to see it processing
        for i in range(15):
            await asyncio.sleep(1)
            with SessionLocal() as db:
                sess = crud.get_session(db, session_id)
                if sess:
                    print(f"  Progress: {sess.processed_frames} frames processed | Status: {sess.status}")
                if sess and sess.status == "failed":
                    print(f"  Error: {sess.error_message}")
                    break
        
        print("\nStopping test session...")
        with SessionLocal() as db:
            crud.update_session(db, session_id, status="stopped")
            
        await task
        print("  Test Completed.")
        
    except Exception as e:
        print(f"  An error occurred during processing: {e}")

if __name__ == "__main__":
    asyncio.run(test_youtube())
