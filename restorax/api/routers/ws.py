"""
WebSocket endpoint for real-time job progress streaming.

GET /ws/jobs/{job_id}/progress

The client connects and receives JSON messages:
  {"job_id": "...", "progress": 0.42, "status": "running"}
  {"job_id": "...", "progress": 1.0,  "status": "completed", "output_path": "..."}
  {"job_id": "...", "progress": 0.0,  "status": "failed", "error": "..."}

Implementation: subscribes to a Redis pub/sub channel published by ProgressReporter.
The connection closes automatically when the job reaches a terminal state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from restorax.config import settings

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "restorax:job_progress:"
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@router.websocket("/ws/jobs/{job_id}/progress")
async def job_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    channel = f"{_CHANNEL_PREFIX}{job_id}"
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(channel)

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            await websocket.send_json(data)
            if data.get("status") in _TERMINAL_STATUSES:
                break
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected for job %s", job_id)
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe(channel)
        await client.aclose()
        with contextlib.suppress(Exception):
            await websocket.close()
