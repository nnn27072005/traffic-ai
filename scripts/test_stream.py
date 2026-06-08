import cv2
import requests
import time
import argparse
import sys
import tempfile
import urllib.request
import os

API_URL = "http://localhost:8000"
VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"

def test_stream(video_source, fps=10):
    print(f"Downloading/Opening video source: {video_source}")
    
    if video_source.startswith("http"):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        print(f"Downloading to {temp_file.name}...")
        urllib.request.urlretrieve(video_source, temp_file.name)
        cap = cv2.VideoCapture(temp_file.name)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Failed to open video")
        sys.exit(1)

    print("Checking backend health...")
    try:
        health = requests.get(f"{API_URL}/health")
        if health.status_code != 200:
            print("Backend not healthy!")
            sys.exit(1)
        print("Backend is healthy. Starting stream simulation...")
    except requests.exceptions.ConnectionError:
        print("Backend connection failed! Make sure the API is running on port 8000.")
        sys.exit(1)

    # We will simulate processing session 1 (which matches the frontend fallback/default for now)
    session_id = 1
    
    frame_delay = 1.0 / fps
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            start_time = time.time()
            
            # Encode frame
            _, encoded_img = cv2.imencode('.jpg', frame)
            
            # Send to API
            files = {'file': ('frame.jpg', encoded_img.tobytes(), 'image/jpeg')}
            
            print(f"Sending frame {frame_count}...", end="\r")
            
            # To avoid overloading the local server and simulate a 10FPS stream
            response = requests.post(
                f"{API_URL}/analyze/image",
                files=files,
                params={"session_id": session_id, "line_ratio": 0.5}
            )
            
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        if video_source.startswith("http"):
            os.remove(temp_file.name)
        print(f"\nFinished sending {frame_count} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate video stream to backend API")
    parser.add_argument("--source", type=str, default=VIDEO_URL, help="Path or URL to video file")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS to send")
    args = parser.parse_args()
    
    test_stream(args.source, args.fps)
