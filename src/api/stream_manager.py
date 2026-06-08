# src/api/stream_manager.py
import cv2
import numpy as np
from typing import Dict, Optional
import asyncio

class StreamManager:
    """
    Manages active video streams and their latest processed frames.
    Allows multiple clients to consume the same stream.
    """
    def __init__(self):
        # session_id -> latest_encoded_frame
        self._frames: Dict[int, bytes] = {}
        # session_id -> Event to notify new frames
        self._events: Dict[int, asyncio.Event] = {}

    def update_frame(self, session_id: int, frame: np.ndarray):
        """Encodes and updates the latest frame for a session."""
        _, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self._frames[session_id] = encoded_img.tobytes()
        
        if session_id not in self._events:
            self._events[session_id] = asyncio.Event()
        
        self._events[session_id].set()
        self._events[session_id].clear()

    async def get_frame_generator(self, session_id: int):
        """Yields multipart JPEG frames for the given session."""
        if session_id not in self._events:
            self._events[session_id] = asyncio.Event()

        while True:
            await self._events[session_id].wait()
            frame = self._frames.get(session_id)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                await asyncio.sleep(0.1)

    def cleanup_session(self, session_id: int):
        self._frames.pop(session_id, None)
        self._events.pop(session_id, None)

# Global instance
stream_manager = StreamManager()
