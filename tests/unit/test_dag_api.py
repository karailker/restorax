"""Tests for DAG pipeline API endpoints."""

from __future__ import annotations

from datetime import UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from restorax.dag.graph import DAG
from restorax.dag.nodes.control import PassNode
from restorax.dag.serializer import DAGSerializer


def _minimal_dag_config() -> dict:
    """Build a valid 1-node DAG config for API testing."""
    dag = DAG(
        id="test-dag",
        name="Test DAG",
        nodes={"p1": PassNode(id="p1", name="Pass")},
        edges=[],
    )
    return DAGSerializer.to_dict(dag)


@pytest.fixture(scope="module")
def client():
    from restorax.api.app import app

    return TestClient(app)


def test_create_dag_returns_201(client):
    config = _minimal_dag_config()
    config["id"] = "api-test-dag-001"

    with patch("restorax.api.routers.pipelines.PipelineRepository") as MockRepo:
        mock_repo = AsyncMock()
        mock_model = MagicMock()
        mock_model.id = "api-test-dag-001"
        mock_model.name = "Test DAG"
        mock_model.description = ""
        mock_model.config = config
        from datetime import datetime

        mock_model.created_at = datetime.now(UTC)
        mock_model.updated_at = datetime.now(UTC)
        mock_repo.create.return_value = mock_model
        MockRepo.return_value = mock_repo

        resp = client.post(
            "/pipelines/dag",
            json={
                "id": "api-test-dag-001",
                "name": "Test DAG",
                "config": config,
            },
        )

    assert resp.status_code == 201
    assert resp.json()["id"] == "api-test-dag-001"


def test_create_dag_invalid_config_returns_422(client):
    resp = client.post(
        "/pipelines/dag",
        json={
            "id": "bad-dag",
            "name": "Bad",
            "config": {
                "schema_type": "dag",
                "id": "bad",
                "name": "bad",
                "nodes": [{"type": "nonexistent_type", "id": "x", "name": "X"}],
                "edges": [],
            },
        },
    )
    assert resp.status_code == 422


def test_get_branches_for_nonexistent_job_returns_404(client):
    with patch("restorax.api.routers.jobs.JobRepository") as MockRepo:
        from restorax.core.exceptions import JobNotFoundError

        mock_repo = AsyncMock()
        mock_repo.get.side_effect = JobNotFoundError("not found")
        MockRepo.return_value = mock_repo
        resp = client.get("/jobs/no-such-job/branches")
    assert resp.status_code == 404


def test_branches_come_from_recorded_node_metadata(client):
    """Branch rows describe what the parallel node ran, not node-id substring matches."""
    with patch("restorax.api.routers.jobs.JobRepository") as MockRepo:
        job = SimpleNamespace(
            dag_run={
                "node_states": {"par": "succeeded"},
                "node_metadata": {
                    "par": {
                        "branches": [
                            {"index": 0, "name": "sharp", "preview_path": None},
                            {"index": 1, "name": "soft", "preview_path": None},
                        ]
                    }
                },
            }
        )
        mock_repo = AsyncMock()
        mock_repo.get.return_value = job
        MockRepo.return_value = mock_repo
        resp = client.get("/jobs/any/branches")

    assert resp.status_code == 200
    branches = resp.json()["branches"]
    assert [b["name"] for b in branches] == ["sharp", "soft"]
    assert {b["node_id"] for b in branches} == {"par"}
    assert all(b["status"] == "succeeded" for b in branches)


def test_branches_empty_without_metadata(client):
    with patch("restorax.api.routers.jobs.JobRepository") as MockRepo:
        mock_repo = AsyncMock()
        mock_repo.get.return_value = SimpleNamespace(dag_run=None)
        MockRepo.return_value = mock_repo
        resp = client.get("/jobs/any/branches")

    assert resp.status_code == 200
    assert resp.json()["branches"] == []
