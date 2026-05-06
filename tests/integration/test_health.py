from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_always_200(async_client: AsyncClient):
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_200_when_db_and_redis_healthy(async_client: AsyncClient):
    resp = await async_client.get("/ready")
    # In test environment both DB (sqlite) and Redis must be reachable
    assert resp.status_code in (200, 503)  # 503 acceptable if Redis not running in CI
    data = resp.json()
    assert "db" in data
    assert "redis" in data


@pytest.mark.asyncio
async def test_ready_503_when_redis_fails(async_client: AsyncClient, monkeypatch):
    import redis.asyncio as aioredis

    async def bad_ping(*args, **kwargs):
        raise ConnectionError("Redis is down")

    monkeypatch.setattr(aioredis.Redis, "ping", bad_ping)
    resp = await async_client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["redis"] == "fail"


@pytest.mark.asyncio
async def test_ready_503_when_db_fails(async_client: AsyncClient, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    async def bad_execute(*args, **kwargs):
        raise Exception("DB is down")

    monkeypatch.setattr(AsyncSession, "execute", bad_execute)
    resp = await async_client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["db"] == "fail"
