"""Tests for DAG pipeline API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from restorax.dag.nodes.control import PassNode
from restorax.dag.serializer import DAGSerializer
from restorax.dag.graph import DAG
from restorax.dag.edge import Edge


def _minimal_dag_config() -> dict:
    """Build a valid 1-node DAG config for API testing."""
    from restorax.dag.nodes.control import PassNode
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
        from datetime import datetime, timezone
        mock_model.created_at = datetime.now(timezone.utc)
        mock_model.updated_at = datetime.now(timezone.utc)
        mock_repo.create.return_value = mock_model
        MockRepo.return_value = mock_repo

        resp = client.post("/pipelines/dag", json={
            "id": "api-test-dag-001",
            "name": "Test DAG",
            "config": config,
        })

    assert resp.status_code == 201
    assert resp.json()["id"] == "api-test-dag-001"


def test_create_dag_invalid_config_returns_422(client):
    resp = client.post("/pipelines/dag", json={
        "id": "bad-dag",
        "name": "Bad",
        "config": {"schema_type": "dag", "id": "bad", "name": "bad", "nodes": [{"type": "nonexistent_type", "id": "x", "name": "X"}], "edges": []},
    })
    assert resp.status_code == 422


def test_get_branches_for_nonexistent_job_returns_404(client):
    with patch("restorax.api.routers.jobs.JobRepository") as MockRepo:
        from restorax.core.exceptions import JobNotFoundError
        mock_repo = AsyncMock()
        mock_repo.get.side_effect = JobNotFoundError("not found")
        MockRepo.return_value = mock_repo
        resp = client.get("/jobs/no-such-job/branches")
    assert resp.status_code == 404


def test_merge_request_schema_validates():
    from restorax.api.schemas.job import MergeRequest
    req = MergeRequest(strategy="select", branch_index=2)
    assert req.strategy == "select"
    assert req.branch_index == 2

    req2 = MergeRequest(strategy="blend")
    assert req2.branch_index == 0  # default
