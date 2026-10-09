from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from restorax.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ProgressEmitter:
    """
    Publishes per-node, per-branch progress events to Redis pub/sub.
    Uses the same channel prefix as ProgressReporter so the existing
    WebSocket layer forwards events to the browser unchanged.
    """

    _CHANNEL_PREFIX = "restorax:job_progress:"

    def __init__(self, job_id: str, redis_url: str) -> None:
        self._job_id = job_id
        import redis as _redis

        self._redis = _redis.from_url(redis_url, decode_responses=True)

    def emit(
        self,
        node_id: str,
        progress: float,
        branch_index: int = 0,
        status: str = "running",
    ) -> None:
        payload = json.dumps(
            {
                "job_id": self._job_id,
                "node_id": node_id,
                "branch_index": branch_index,
                "progress": round(progress, 4),
                "status": status,
            }
        )
        try:
            self._redis.publish(f"{self._CHANNEL_PREFIX}{self._job_id}", payload)
        except Exception as exc:
            logger.warning("ProgressEmitter publish failed: %s", exc)


@dataclass
class ExecutionContext:
    """Per-run context passed to every node.execute() call."""

    run_id: str
    job_id: str
    work_dir: Path
    device: torch.device
    registry: ModelRegistry
    progress_emitter: ProgressEmitter
    logger: Any  # structlog.BoundLogger
    config: dict[str, Any] = field(default_factory=dict)
