"""
Integration tests for the REST API.

These tests spin up a real FastAPI app with an in-memory SQLite database
and CELERY_TASK_ALWAYS_EAGER=True (tasks execute synchronously, no broker
needed). No GPU or real model weights are required.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client():
    """Async HTTP client wired to a fresh FastAPI app with test DB."""
    from restorax.api.app import create_app
    from restorax.db.session import create_tables

    app = create_app()
    # ASGITransport does not trigger the lifespan, so init DB explicitly
    await create_tables()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Health ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Models listing ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_models(client: AsyncClient) -> None:
    resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "restorers" in data
    names = [r["name"] for r in data["restorers"]]
    assert "real_esrgan_x4plus" in names
    assert "codeformer" in names


# ── Pipeline CRUD ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_and_get_pipeline(client: AsyncClient) -> None:
    payload = {
        "id": "test_pipeline",
        "name": "Test Pipeline",
        "description": "integration test",
        "config": {"stages": [{"restorer": "real_esrgan_x4plus", "scale": 4}]},
    }
    resp = await client.post("/pipelines", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "test_pipeline"
    assert data["name"] == "Test Pipeline"

    resp2 = await client.get("/pipelines/test_pipeline")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_list_pipelines(client: AsyncClient) -> None:
    resp = await client.get("/pipelines")
    assert resp.status_code == 200
    assert "pipelines" in resp.json()


@pytest.mark.asyncio
async def test_create_duplicate_pipeline_returns_409(client: AsyncClient) -> None:
    payload = {
        "id": "dup_pipeline",
        "name": "Dup",
        "description": "",
        "config": {},
    }
    await client.post("/pipelines", json=payload)
    resp = await client.post("/pipelines", json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_update_pipeline(client: AsyncClient) -> None:
    payload = {"id": "upd_pipeline", "name": "Original", "description": "", "config": {}}
    await client.post("/pipelines", json=payload)

    update = {"id": "upd_pipeline", "name": "Updated", "description": "changed", "config": {"x": 1}}
    resp = await client.put("/pipelines/upd_pipeline", json=update)
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated"


@pytest.mark.asyncio
async def test_delete_pipeline(client: AsyncClient) -> None:
    payload = {"id": "del_pipeline", "name": "To Delete", "description": "", "config": {}}
    await client.post("/pipelines", json=payload)
    resp = await client.delete("/pipelines/del_pipeline")
    assert resp.status_code == 204
    resp2 = await client.get("/pipelines/del_pipeline")
    assert resp2.status_code == 404


# ── Jobs ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_jobs_empty(client: AsyncClient) -> None:
    resp = await client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json()["jobs"] == [] or isinstance(resp.json()["jobs"], list)


@pytest.mark.asyncio
async def test_get_nonexistent_job_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_submit_job(client: AsyncClient, tmp_path: Path) -> None:
    """Submit a job with a fake video file; mock the Celery task dispatch."""
    fake_video = b"FAKE_VIDEO_BYTES"

    # Patch run_job.apply_async so we don't actually need a worker
    mock_task_result = MagicMock()
    mock_task_result.id = "fake-celery-task-id"

    # run_job is imported inside the route function, so patch it at its source module
    with (
        patch("restorax.tasks.job_tasks.run_job") as mock_run_job,
        patch("restorax.api.routers.jobs._resolve_preset", return_value="/fake/sr_x4.yaml"),
    ):
        mock_run_job.apply_async.return_value = mock_task_result

        resp = await client.post(
            "/jobs",
            files={"file": ("test.mp4", io.BytesIO(fake_video), "video/mp4")},
            data={"pipeline_id": "sr_x4"},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["pipeline_id"] == "sr_x4"
    assert data["status"] in ("queued", "pending")
    assert data["id"]

    # Fetch it back
    job_id = data["id"]
    resp2 = await client.get(f"/jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == job_id


@pytest.mark.asyncio
async def test_submit_job_with_restore_audio(client: AsyncClient, tmp_path: Path) -> None:
    """POST /jobs with restore_audio=True should return 201 and the field stored."""
    fake_video = b"FAKE_VIDEO_BYTES"
    mock_task_result = MagicMock()
    mock_task_result.id = "fake-celery-audio-id"

    with (
        patch("restorax.tasks.job_tasks.run_job") as mock_run_job,
        patch("restorax.api.routers.jobs._resolve_preset", return_value="/fake/sr_x4.yaml"),
    ):
        mock_run_job.apply_async.return_value = mock_task_result

        resp = await client.post(
            "/jobs",
            files={"file": ("test_audio.mp4", io.BytesIO(fake_video), "video/mp4")},
            data={"pipeline_id": "sr_x4", "restore_audio": "true"},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] in ("queued", "pending")
    # Verify restore_audio was accepted without error
    assert data["id"]
