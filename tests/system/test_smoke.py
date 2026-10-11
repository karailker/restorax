"""
System smoke tests — run the full FastAPI application (lifespan + DB init)
and verify the key API endpoints work end-to-end without any mocks.

These tests skip automatically when a live Redis instance is unavailable,
so they pass in CI without the broker.  Run them locally with:

    docker compose -f docker-compose.deps.yml up -d
    pytest tests/system/ -v
"""

from __future__ import annotations

import io
import socket
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


def _redis_available() -> bool:
    try:
        s = socket.create_connection(("localhost", 6379), timeout=0.5)
        s.close()
        return True
    except OSError:
        return False


@pytest_asyncio.fixture
async def live_client():
    """Start the full app including lifespan (DB tables created by lifespan)."""
    from restorax.api.app import create_app
    from restorax.db.session import create_tables

    app = create_app()
    await create_tables()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Core health & schema ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health(live_client):
    resp = await live_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_openapi_json(live_client):
    resp = await live_client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    # Verify main resource paths are documented
    assert "/jobs" in schema["paths"]
    assert "/models" in schema["paths"]
    assert "/pipelines" in schema["paths"]


# ── Models listing ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_models_endpoint_complete(live_client):
    resp = await live_client.get("/models")
    assert resp.status_code == 200
    restorers = resp.json()["restorers"]
    assert len(restorers) >= 20, f"Expected ≥20 restorers, got {len(restorers)}"

    # Spot-check categories are present
    categories = {r["category"] for r in restorers}
    assert "super_resolution" in categories
    assert "face_restoration" in categories
    assert "frame_interpolation" in categories
    assert "colorization" in categories


# ── Pipeline CRUD full cycle ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_full_crud(live_client):
    pid = "smoke_pipeline_full"

    # Create
    r = await live_client.post(
        "/pipelines", json={"id": pid, "name": "Smoke Test", "description": "e2e", "config": {}}
    )
    assert r.status_code == 201

    # Read
    r = await live_client.get(f"/pipelines/{pid}")
    assert r.status_code == 200
    assert r.json()["description"] == "e2e"

    # Update
    r = await live_client.put(
        f"/pipelines/{pid}",
        json={"id": pid, "name": "Updated", "description": "updated", "config": {"x": 1}},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "Updated"

    # Delete
    r = await live_client.delete(f"/pipelines/{pid}")
    assert r.status_code == 204

    # Confirm gone
    r = await live_client.get(f"/pipelines/{pid}")
    assert r.status_code == 404


# ── Job submission cycle ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_job_submit_fetch_delete(live_client):
    """Submit a job (mocked task dispatch), fetch it, then delete it."""
    mock_result = MagicMock()
    mock_result.id = "smoke-celery-id"

    with (
        patch("restorax.tasks.job_tasks.run_job") as mock_run,
        patch("restorax.api.routers.jobs._resolve_preset", return_value="/fake/sr_x4.yaml"),
    ):
        mock_run.apply_async.return_value = mock_result

        resp = await live_client.post(
            "/jobs",
            files={"file": ("smoke.mp4", io.BytesIO(b"SMOKE"), "video/mp4")},
            data={"pipeline_id": "sr_x4"},
        )

    assert resp.status_code == 201
    job = resp.json()
    assert job["status"] in ("queued", "pending")
    job_id = job["id"]

    # Fetch
    r = await live_client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["id"] == job_id

    # Job appears in list
    r = await live_client.get("/jobs?limit=50")
    ids = [j["id"] for j in r.json()["jobs"]]
    assert job_id in ids

    # Delete
    r = await live_client.delete(f"/jobs/{job_id}")
    assert r.status_code == 204

    # Confirm deleted
    r = await live_client.get(f"/jobs/{job_id}")
    assert r.status_code == 404


# ── Input validation ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_job_invalid_inputs(live_client):
    # No file
    r = await live_client.post("/jobs", data={"pipeline_id": "sr_x4"})
    assert r.status_code == 422

    # No pipeline_id
    r = await live_client.post(
        "/jobs",
        files={"file": ("v.mp4", io.BytesIO(b"x"), "video/mp4")},
    )
    assert r.status_code == 422

    # GET on nonexistent job
    r = await live_client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
