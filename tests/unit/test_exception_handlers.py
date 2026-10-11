"""Tests that RestoraXError subclasses map to correct HTTP status + body."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from restorax.core.exceptions import (
    JobNotFoundError,
    PipelineConfigError,
    RestoraXError,
    RestorerLoadError,
    RestorerNotFoundError,
)


def _app_with_probe_routes() -> FastAPI:
    """Create a fresh app instance with one probe route per exception type."""
    from restorax.api.app import create_app

    app = create_app()

    @app.get("/probe/restorer-load-error")
    async def _raise_load():
        raise RestorerLoadError("weights missing")

    @app.get("/probe/restorer-not-found")
    async def _raise_not_found():
        raise RestorerNotFoundError("unknown_model")

    @app.get("/probe/job-not-found")
    async def _raise_job():
        raise JobNotFoundError("job-123")

    @app.get("/probe/pipeline-config-error")
    async def _raise_pipeline():
        raise PipelineConfigError("bad yaml")

    @app.get("/probe/base-error")
    async def _raise_base():
        raise RestoraXError("something broke")

    return app


@pytest.fixture(scope="module")
def client():
    return TestClient(_app_with_probe_routes(), raise_server_exceptions=False)


def test_restorer_load_error_returns_503(client):
    resp = client.get("/probe/restorer-load-error")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "restorer_load_error"
    assert "weights missing" in body["message"]


def test_restorer_not_found_returns_404(client):
    resp = client.get("/probe/restorer-not-found")
    assert resp.status_code == 404
    assert resp.json()["error"] == "restorer_not_found"


def test_job_not_found_returns_404(client):
    resp = client.get("/probe/job-not-found")
    assert resp.status_code == 404
    assert resp.json()["error"] == "job_not_found"


def test_pipeline_config_error_returns_422(client):
    resp = client.get("/probe/pipeline-config-error")
    assert resp.status_code == 422
    assert resp.json()["error"] == "pipeline_config_error"


def test_base_restorax_error_returns_500(client):
    resp = client.get("/probe/base-error")
    assert resp.status_code == 500
    assert resp.json()["error"] == "internal_error"
