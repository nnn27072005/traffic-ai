# src/api/routers/stream.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.stream_manager import stream_manager

router = APIRouter(prefix="/stream", tags=["Streaming"])

@router.get("/live/{session_id}")
async def stream_session(session_id: int):
    """
    Returns an MJPEG stream for the processed frames of a specific session.
    """
    return StreamingResponse(
        stream_manager.get_frame_generator(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
