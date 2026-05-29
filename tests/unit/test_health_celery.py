"""Tests for GET /health/celery — Celery queue depth and worker info."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from restorax.api.app import app
    return TestClient(app)


def test_celery_health_returns_expected_fields(client):
    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {"worker1@host": [{"id": "task-1"}]}
    mock_inspect.reserved.return_value = {"worker1@host": [{"id": "task-2"}, {"id": "task-3"}]}

    with patch("restorax.api.routers.health._celery_inspect", return_value=mock_inspect):
        resp = client.get("/health/celery")

    assert resp.status_code == 200
    body = resp.json()
    assert body["workers"] == 1
    assert body["active_tasks"] == 1
    assert body["queued_tasks"] == 2
    assert body["status"] in ("ok", "degraded", "unavailable")


def test_celery_health_unavailable_when_celery_unreachable(client):
    mock_inspect = MagicMock()
    mock_inspect.active.return_value = None  # Celery returns None when no workers

    with patch("restorax.api.routers.health._celery_inspect", return_value=mock_inspect):
        resp = client.get("/health/celery")

    assert resp.status_code == 200
    assert resp.json()["status"] == "unavailable"
    assert resp.json()["workers"] == 0
