"""
ProgressReporter: publishes job progress events to a Redis pub/sub channel.

The FastAPI WebSocket router subscribes to the channel and streams events
to the browser client in real time.
"""

from __future__ import annotations

import json
import logging

import redis

from restorax.config import settings

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "restorax:job_progress:"


def _get_redis() -> redis.Redis:  # type: ignore[type-arg]
    return redis.from_url(settings.redis_url, decode_responses=True)


class ProgressReporter:
    """Publishes progress events for a single job."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self._channel = f"{_CHANNEL_PREFIX}{job_id}"
        self._redis = _get_redis()

    def update(self, progress: float, status: str = "running") -> None:
        payload = json.dumps(
            {"job_id": self._job_id, "progress": round(progress, 4), "status": status}
        )
        try:
            self._redis.publish(self._channel, payload)
        except Exception as exc:
            logger.warning("Failed to publish progress for job %s: %s", self._job_id, exc)

    def complete(self, output_path: str) -> None:
        payload = json.dumps(
            {
                "job_id": self._job_id,
                "progress": 1.0,
                "status": "completed",
                "output_path": output_path,
            }
        )
        self._redis.publish(self._channel, payload)

    def fail(self, error: str) -> None:
        payload = json.dumps(
            {"job_id": self._job_id, "progress": 0.0, "status": "failed", "error": error}
        )
        self._redis.publish(self._channel, payload)

    def channel(self) -> str:
        return self._channel
