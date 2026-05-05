from __future__ import annotations

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app() -> FastAPI:
    from restorax.api.middleware import RequestIDMiddleware
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    async def ping():
        ctx = structlog.contextvars.get_contextvars()
        return {"request_id": ctx.get("request_id")}

    return app


def test_request_id_generated_when_absent():
    client = TestClient(_make_app())
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    rid = resp.headers["X-Request-ID"]
    assert len(rid) == 36  # uuid4 format


def test_request_id_propagated_from_header():
    client = TestClient(_make_app())
    resp = client.get("/ping", headers={"X-Request-ID": "my-custom-id"})
    assert resp.headers["X-Request-ID"] == "my-custom-id"


def test_request_id_bound_to_structlog_context():
    client = TestClient(_make_app())
    resp = client.get("/ping", headers={"X-Request-ID": "trace-abc"})
    assert resp.json()["request_id"] == "trace-abc"


def test_context_cleared_between_requests():
    """Structlog context from request N must not leak into request N+1."""
    client = TestClient(_make_app())
    resp1 = client.get("/ping", headers={"X-Request-ID": "req-1"})
    resp2 = client.get("/ping", headers={"X-Request-ID": "req-2"})
    assert resp1.json()["request_id"] == "req-1"
    assert resp2.json()["request_id"] == "req-2"
