"""
Round-robin GPU queue router for Celery.

When multiple GPU workers are running (e.g. RESTORAX_GPU_QUEUES=gpu_0,gpu_1),
each job is dispatched to the next available GPU queue in rotation.
Single-GPU setups always route to gpu_default.

Usage:
    # .env or environment
    RESTORAX_GPU_QUEUES=gpu_0,gpu_1

    # Celery workers (one per GPU)
    CUDA_VISIBLE_DEVICES=0 celery worker --queues gpu_0 --concurrency=1
    CUDA_VISIBLE_DEVICES=1 celery worker --queues gpu_1 --concurrency=1
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterator

_DEFAULT_QUEUE = "gpu_default"


def _build_queue_cycle() -> Iterator[str]:
    raw = os.environ.get("RESTORAX_GPU_QUEUES", "").strip()
    queues = [q.strip() for q in raw.split(",") if q.strip()] if raw else [_DEFAULT_QUEUE]
    return itertools.cycle(queues)


_queue_cycle = _build_queue_cycle()


def next_gpu_queue() -> str:
    """Return the next GPU queue name in the round-robin rotation."""
    return next(_queue_cycle)


def reset_router() -> None:
    """Re-read RESTORAX_GPU_QUEUES and reset the cycle (useful in tests)."""
    global _queue_cycle
    _queue_cycle = _build_queue_cycle()
