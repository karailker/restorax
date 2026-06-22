# Backend Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the RestoraX backend before the Pipeline DAG and UI work — fix three specific gaps found in the audit while leaving everything else unchanged.

**Architecture:** Three targeted patches: (1) add audio restorers to the `/models` list endpoint, (2) add global FastAPI exception handlers so `RestoraXError` subclasses return structured JSON instead of 500s, (3) add a `/health/celery` sub-endpoint exposing queue depth and worker count. No refactoring beyond these three gaps.

**Tech Stack:** FastAPI exception handlers, Celery Inspect API (already a dependency), pytest with httpx AsyncClient.

**Audit findings (do not re-investigate):**
- `restorax/api/routers/models.py:29-37` — 21 restorers listed; audio 3 (Demucs, VoiceFixer, RNNoise) missing
- `restorax/api/app.py` — no `@app.exception_handler()` or `add_exception_handler()` calls
- `restorax/api/routers/health.py` — `/health` returns `{"status":"ok"}`, `/ready` checks DB+Redis only; no Celery metrics

---

## File Map

| File | Change |
|---|---|
| `restorax/api/routers/models.py` | Add 3 audio restorer imports + entries in `_RESTORER_CLASSES` |
| `restorax/api/app.py` | Register exception handlers for `RestorerLoadError`, `RestorerNotFoundError`, `JobNotFoundError`, `PipelineConfigError`, `RestoraXError` |
| `restorax/api/routers/health.py` | Add `GET /health/celery` → queue depth + worker count |
| `tests/unit/test_models_router.py` | New — verify audio restorers appear in response |
| `tests/unit/test_exception_handlers.py` | New — verify each exception maps to correct HTTP status + body |
| `tests/unit/test_health_celery.py` | New — verify Celery endpoint structure |

---

## Task 1 — Add audio restorers to `/models`

**Files:**
- Modify: `restorax/api/routers/models.py`
- Create: `tests/unit/test_models_router.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_models_router.py`:

```python
"""Tests for GET /models — verifies all restorer categories are present."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from restorax.api.app import app
    return TestClient(app)


def test_models_includes_audio_restorers(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    names = {r["name"] for r in resp.json()["restorers"]}
    assert "demucs" in names, f"demucs missing — got {names}"
    assert "voicefixer" in names, f"voicefixer missing — got {names}"
    assert "rnnoise" in names, f"rnnoise missing — got {names}"


def test_models_lists_all_categories(client):
    resp = client.get("/models")
    categories = {r["category"] for r in resp.json()["restorers"]}
    expected = {
        "super_resolution", "face_restoration", "colorization",
        "frame_interpolation", "artifact_removal", "hdr",
        "stabilization", "deinterlacing", "audio",
    }
    assert expected <= categories, f"Missing categories: {expected - categories}"


def test_models_response_fields(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    for r in resp.json()["restorers"]:
        assert "name" in r
        assert "category" in r
        assert "scale_factor" in r
        assert "min_vram_gb" in r
        assert "tags" in r
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n restorax python -m pytest tests/unit/test_models_router.py::test_models_includes_audio_restorers -x -q
```

Expected: FAIL — `demucs missing`

- [ ] **Step 3: Add audio restorers to models.py**

In `restorax/api/routers/models.py`, after the existing imports (line 25), add:

```python
from restorax.restorers.audio.demucs import DemucsRestorer
from restorax.restorers.audio.voicefixer import VoiceFixerRestorer
from restorax.restorers.audio.rnnoise import RNNoiseRestorer
```

Then add to `_RESTORER_CLASSES` (line 36, before the closing `]`):

```python
    DemucsRestorer, VoiceFixerRestorer, RNNoiseRestorer,
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n restorax python -m pytest tests/unit/test_models_router.py -x -q
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/api/routers/models.py tests/unit/test_models_router.py
git commit -m "feat: expose audio restorers in GET /models endpoint

Demucs, VoiceFixer, and RNNoise were registered only in job_tasks.py
but missing from the /models list API. Now all 24 restorers are listed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2 — Global exception handlers for RestoraXError

**Files:**
- Modify: `restorax/api/app.py`
- Create: `tests/unit/test_exception_handlers.py`

**Mapping (error class → HTTP status → error code string):**
| Exception | Status | `"error"` field |
|---|---|---|
| `RestorerLoadError` | 503 | `"restorer_load_error"` |
| `RestorerNotFoundError` | 404 | `"restorer_not_found"` |
| `JobNotFoundError` | 404 | `"job_not_found"` |
| `PipelineConfigError` | 422 | `"pipeline_config_error"` |
| `RestoraXError` (base) | 500 | `"internal_error"` |

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_exception_handlers.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n restorax python -m pytest tests/unit/test_exception_handlers.py -x -q
```

Expected: FAIL — all return 500 (no handlers yet)

- [ ] **Step 3: Add exception handlers in app.py**

In `restorax/api/app.py`, after the existing imports add:

```python
from fastapi import Request
from fastapi.responses import JSONResponse
```

Inside `create_app()`, after the router includes and before the `/metrics` route, add:

```python
    # ── Exception handlers ────────────────────────────────────────────────────
    from restorax.core.exceptions import (
        JobNotFoundError,
        PipelineConfigError,
        RestoraXError,
        RestorerLoadError,
        RestorerNotFoundError,
    )

    @app.exception_handler(RestorerLoadError)
    async def _handle_restorer_load_error(request: Request, exc: RestorerLoadError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "restorer_load_error", "message": str(exc)})

    @app.exception_handler(RestorerNotFoundError)
    async def _handle_restorer_not_found(request: Request, exc: RestorerNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "restorer_not_found", "message": str(exc)})

    @app.exception_handler(JobNotFoundError)
    async def _handle_job_not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "job_not_found", "message": str(exc)})

    @app.exception_handler(PipelineConfigError)
    async def _handle_pipeline_config_error(request: Request, exc: PipelineConfigError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "pipeline_config_error", "message": str(exc)})

    @app.exception_handler(RestoraXError)
    async def _handle_restorax_error(request: Request, exc: RestoraXError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(exc)})
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n restorax python -m pytest tests/unit/test_exception_handlers.py -x -q
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/api/app.py tests/unit/test_exception_handlers.py
git commit -m "feat: add global exception handlers for RestoraXError subclasses

RestorerLoadError → 503, RestorerNotFoundError/JobNotFoundError → 404,
PipelineConfigError → 422, RestoraXError base → 500. All return
{\"error\": \"<code>\", \"message\": \"<detail>\"} JSON bodies.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3 — Celery queue metrics in `/health/celery`

**Files:**
- Modify: `restorax/api/routers/health.py`
- Create: `tests/unit/test_health_celery.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_health_celery.py`:

```python
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
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n restorax python -m pytest tests/unit/test_health_celery.py -x -q
```

Expected: FAIL — `/health/celery` route does not exist

- [ ] **Step 3: Add `/health/celery` to health.py**

Add at the bottom of `restorax/api/routers/health.py`:

```python
def _celery_inspect():
    """Return a Celery Inspect instance. Extracted for testability."""
    from restorax.tasks.celery_app import celery_app
    return celery_app.control.inspect(timeout=2.0)


@router.get("/health/celery")
async def celery_health() -> dict:
    """Return Celery worker count, active task count, and queued task count."""
    try:
        inspect = _celery_inspect()
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
    except Exception:
        return {"status": "unavailable", "workers": 0, "active_tasks": 0, "queued_tasks": 0}

    if not active and not reserved:
        return {"status": "unavailable", "workers": 0, "active_tasks": 0, "queued_tasks": 0}

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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n restorax python -m pytest tests/unit/test_health_celery.py -x -q
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/api/routers/health.py tests/unit/test_health_celery.py
git commit -m "feat: add GET /health/celery endpoint with queue depth and worker count

Returns {status, workers, active_tasks, queued_tasks}. Uses Celery
Inspect API with 2s timeout. Returns status=unavailable when no
workers respond rather than erroring.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4 — Push and verify full test suite

- [ ] **Step 1: Run all new tests together**

```bash
conda run -n restorax python -m pytest \
  tests/unit/test_models_router.py \
  tests/unit/test_exception_handlers.py \
  tests/unit/test_health_celery.py \
  -v
```

Expected: 10 passed, 0 failed

- [ ] **Step 2: Run existing unit tests to confirm no regressions**

```bash
conda run -n restorax python -m pytest tests/unit/ -q --tb=short
```

Expected: all passing (skip count may increase due to torch/weight guards — that's fine)

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Completion Criteria

- [ ] `GET /models` returns all 24 restorers including demucs, voicefixer, rnnoise
- [ ] `RestorerLoadError` raised in any route returns `{"error": "restorer_load_error", ...}` with HTTP 503
- [ ] `RestorerNotFoundError` / `JobNotFoundError` return HTTP 404
- [ ] `PipelineConfigError` returns HTTP 422
- [ ] `GET /health/celery` returns `{status, workers, active_tasks, queued_tasks}`
- [ ] 10 new tests pass, no existing tests broken
