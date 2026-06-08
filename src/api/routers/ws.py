from fastapi import APIRouter, WebSocket

from src.api.websocket_manager import websocket_manager

router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/sessions/{session_id}")
async def session_events(session_id: int, websocket: WebSocket):
    await websocket_manager.connect(session_id, websocket)
    await websocket_manager.listen_until_disconnect(session_id, websocket)
