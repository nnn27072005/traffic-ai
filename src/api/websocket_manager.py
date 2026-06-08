from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger("traffic-api.websocket")


class WebSocketManager:
    """Keeps WebSocket subscribers grouped by processing session."""

    def __init__(self) -> None:
        self._connections: dict[int, set[WebSocket]] = defaultdict(set)

    async def connect(self, session_id: int, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[session_id].add(websocket)
        logger.info("WebSocket connected: session=%s clients=%s", session_id, len(self._connections[session_id]))

    def disconnect(self, session_id: int, websocket: WebSocket) -> None:
        connections = self._connections.get(session_id)
        if not connections:
            return
        connections.discard(websocket)
        if not connections:
            self._connections.pop(session_id, None)
        logger.info("WebSocket disconnected: session=%s", session_id)

    async def broadcast(self, session_id: int, payload: dict[str, Any]) -> None:
        connections = list(self._connections.get(session_id, ()))
        if not connections:
            return

        stale: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except WebSocketDisconnect:
                stale.append(websocket)
            except Exception:
                logger.exception("Failed to send WebSocket payload for session %s", session_id)
                stale.append(websocket)

        for websocket in stale:
            self.disconnect(session_id, websocket)

    async def listen_until_disconnect(self, session_id: int, websocket: WebSocket) -> None:
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            self.disconnect(session_id, websocket)


websocket_manager = WebSocketManager()
