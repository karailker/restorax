"""Extended integration tests: pagination, batch jobs, delete, cancel, WebSocket."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client():
    from restorax.api.app import create_app
    from restorax.db.session import create_tables

    app = create_app()
    await create_tables()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _fake_task():
    mock = MagicMock()
    mock.id = "fake-task-id"
    return mock


# ── Health & OpenAPI ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body or "status" in body


@pytest.mark.asyncio
async def test_openapi_schema_accessible(client):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/jobs" in schema["paths"]


@pytest.mark.asyncio
async def test_docs_accessible(client):
    resp = await client.get("/docs")
    assert resp.status_code == 200


# ── Jobs — list and pagination ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_jobs_returns_list(client):
    resp = await client.get("/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json()["jobs"], list)


@pytest.mark.asyncio
async def test_list_jobs_limit_param(client):
    resp = await client.get("/jobs?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()["jobs"]) <= 1


@pytest.mark.asyncio
async def test_list_jobs_offset_param(client):
    resp = await client.get("/jobs?offset=0&limit=10")
    assert resp.status_code == 200


# ── Jobs — CRUD ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_job_404(client):
    resp = await client.get("/jobs/00000000-0000-0000-0000-000000000001")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_submit_job_returns_201(client):
    with (
        patch("restorax.tasks.job_tasks.run_job") as mock_run,
        patch("restorax.api.routers.jobs._resolve_preset", return_value="/fake/sr_x4.yaml"),
    ):
        mock_run.apply_async.return_value = _fake_task()
        resp = await client.post(
            "/jobs",
            files={"file": ("v.mp4", io.BytesIO(b"fake"), "video/mp4")},
            data={"pipeline_id": "sr_x4"},
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["pipeline_id"] == "sr_x4"
    assert data["status"] in ("queued", "pending")


@pytest.mark.asyncio
async def test_submit_job_missing_file_returns_422(client):
    resp = await client.post("/jobs", data={"pipeline_id": "sr_x4"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_submit_job_missing_pipeline_returns_422(client):
    resp = await client.post(
        "/jobs",
        files={"file": ("v.mp4", io.BytesIO(b"fake"), "video/mp4")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_delete_job_204(client):
    with (
        patch("restorax.tasks.job_tasks.run_job") as mock_run,
        patch("restorax.api.routers.jobs._resolve_preset", return_value="/fake/sr_x4.yaml"),
    ):
        mock_run.apply_async.return_value = _fake_task()
        create_resp = await client.post(
            "/jobs",
            files={"file": ("del.mp4", io.BytesIO(b"d"), "video/mp4")},
            data={"pipeline_id": "sr_x4"},
        )
    job_id = create_resp.json()["id"]
    resp = await client.delete(f"/jobs/{job_id}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_nonexistent_job_404(client):
    resp = await client.delete("/jobs/00000000-0000-0000-0000-000000000099")
    assert resp.status_code == 404


# ── Jobs — batch ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_submit_returns_list(client):
    with (
        patch("restorax.tasks.job_tasks.run_job") as mock_run,
        patch("restorax.api.routers.jobs._resolve_preset", return_value="/fake/sr_x4.yaml"),
    ):
        mock_run.apply_async.return_value = _fake_task()
        resp = await client.post(
            "/jobs/batch",
            files=[
                ("files", ("a.mp4", io.BytesIO(b"a"), "video/mp4")),
                ("files", ("b.mp4", io.BytesIO(b"b"), "video/mp4")),
            ],
            data={"pipeline_id": "sr_x4"},
        )
    assert resp.status_code == 201
    jobs = resp.json()["jobs"]
    assert len(jobs) == 2
    assert all(j["pipeline_id"] == "sr_x4" for j in jobs)


@pytest.mark.asyncio
async def test_batch_submit_empty_files_returns_422(client):
    resp = await client.post("/jobs/batch", data={"pipeline_id": "sr_x4"})
    assert resp.status_code == 422


# ── Models endpoint ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_models_lists_all_restorers(client):
    resp = await client.get("/models")
    assert resp.status_code == 200
    names = [r["name"] for r in resp.json()["restorers"]]
    assert "real_esrgan_x4plus" in names
    assert "mamba_ir_x4" in names
    assert "vrt_x4" in names
    assert "rife_v4" in names
    assert "codeformer" in names


@pytest.mark.asyncio
async def test_models_response_has_required_fields(client):
    resp = await client.get("/models")
    for r in resp.json()["restorers"]:
        assert "name" in r
        assert "category" in r
        assert "min_vram_gb" in r
        assert "scale_factor" in r


# ── Pipelines CRUD ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_not_found_returns_404(client):
    resp = await client.get("/pipelines/does_not_exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_pipeline_and_retrieve(client):
    payload = {"id": "ext_test_pl", "name": "Ext Test", "description": "", "config": {}}
    r1 = await client.post("/pipelines", json=payload)
    assert r1.status_code == 201
    r2 = await client.get("/pipelines/ext_test_pl")
    assert r2.status_code == 200
    assert r2.json()["name"] == "Ext Test"
