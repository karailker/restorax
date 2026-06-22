from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> JSONResponse:
    db_status = await _check_db()
    redis_status = await _check_redis()
    body = {"db": db_status, "redis": redis_status}
    status_code = 200 if db_status == "ok" and redis_status == "ok" else 503
    return JSONResponse(content=body, status_code=status_code)


async def _check_db() -> str:
    try:
        from sqlalchemy import text
        from restorax.db.session import AsyncSessionLocal
        async with asyncio.timeout(2.0):
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "fail"


async def _check_redis() -> str:
    try:
        import redis.asyncio as aioredis
        from restorax.config import settings
        async with asyncio.timeout(2.0):
            r = aioredis.from_url(settings.redis_url)
            await r.ping()
            await r.aclose()
        return "ok"
    except Exception:
        return "fail"


def _celery_inspect():
    """Return a Celery Inspect instance. Extracted for testability."""
    from restorax.tasks.celery_app import celery_app
    return celery_app.control.inspect(timeout=2.0)


@router.get("/health/celery")
async def celery_health() -> dict:
    """Return Celery worker count, active task count, and queued task count."""
    try:
        inspect = _celery_inspect()
        active_raw = await asyncio.to_thread(inspect.active)
        reserved_raw = await asyncio.to_thread(inspect.reserved)
    except Exception:
        return {"status": "unavailable", "workers": 0, "active_tasks": 0, "queued_tasks": 0}

    if active_raw is None:
        return {"status": "unavailable", "workers": 0, "active_tasks": 0, "queued_tasks": 0}

    active = active_raw or {}
    reserved = reserved_raw or {}

    workers = len(set(list(active.keys()) + list(reserved.keys())))
    active_tasks = sum(len(tasks) for tasks in active.values())
    queued_tasks = sum(len(tasks) for tasks in reserved.values())
    status = "ok" if workers > 0 else "degraded"

    return {
        "status": status,
        "workers": workers,
        "active_tasks": active_tasks,
        "queued_tasks": queued_tasks,
    }
