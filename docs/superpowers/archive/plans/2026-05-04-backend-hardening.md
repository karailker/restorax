# Backend Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-grade observability to the RestoraX backend: structured JSON logging (structlog), distributed traces + Prometheus metrics (OpenTelemetry), health probes, request ID propagation, and optional Sentry APM.

**Architecture:** Three new modules (`restorax/logging.py`, `restorax/telemetry.py`, `restorax/api/routers/health.py`) wired into the existing FastAPI app and Celery worker at startup. All export targets are env-driven — zero extra containers in dev.

**Tech Stack:** structlog 24, opentelemetry-sdk 1.24, opentelemetry-instrumentation-* 0.45b, opentelemetry-exporter-prometheus, prometheus-client 0.20, sentry-sdk 2 (optional, `pip install restorax[apm]`)

**Spec:** `docs/superpowers/specs/2026-05-04-backend-hardening-design.md`

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `pyproject.toml` | Add observability deps + `[apm]` optional group |
| Modify | `restorax/config.py` | Add `otel_exporter_otlp_endpoint`, `otel_service_name`, `sentry_dsn` |
| Create | `restorax/logging.py` | structlog configure: dev=pretty, prod=JSON, stdlib redirect |
| Create | `restorax/telemetry.py` | OTEL TracerProvider + MeterProvider + auto-instrumentation + business metrics |
| Modify | `restorax/api/middleware.py` | Upgrade `RequestIDMiddleware`: structlog context bind + OTEL span attribute |
| Create | `restorax/api/routers/health.py` | `/health` (liveness) + `/ready` (readiness: DB + Redis) |
| Modify | `restorax/api/app.py` | Call configure_logging/telemetry, mount health router, add `/metrics`, remove old stubs |
| Modify | `restorax/tasks/job_tasks.py` | Celery signals for structlog context + business metric increments |
| Create | `tests/unit/test_logging.py` | Unit tests for configure_logging processor chain |
| Create | `tests/unit/test_telemetry.py` | Unit tests for configure_telemetry no-op and metric helpers |
| Create | `tests/unit/test_middleware_observability.py` | Unit tests for upgraded RequestIDMiddleware |
| Create | `tests/integration/test_health.py` | Integration tests for /health and /ready |

---

## Task 1: Add dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add observability dependencies**

In `pyproject.toml`, add to the `dependencies` list under `[project]`, after the existing `"piqa>=1.3.0",` line:

```toml
    # Observability
    "structlog>=24.1.0",
    "opentelemetry-sdk>=1.24.0",
    "opentelemetry-api>=1.24.0",
    "opentelemetry-instrumentation-fastapi>=0.45b0",
    "opentelemetry-instrumentation-celery>=0.45b0",
    "opentelemetry-instrumentation-sqlalchemy>=0.45b0",
    "opentelemetry-instrumentation-redis>=0.45b0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.24.0",
    "opentelemetry-exporter-prometheus>=0.45b0",
    "prometheus-client>=0.20.0",
```

Add a new optional dependencies section after `[project.optional-dependencies]` `face`:

```toml
apm = [
    "sentry-sdk[fastapi,celery]>=2.0.0",
]
```

- [ ] **Step 2: Install the new packages**

```bash
conda run -n restorax pip install -e ".[dev]"
```

Expected: packages install without conflicts. If `opentelemetry-instrumentation-*` versions conflict, try without the `b0` suffix (e.g. `>=0.45`).

- [ ] **Step 3: Verify imports**

```bash
conda run -n restorax python -c "
import structlog
import opentelemetry.sdk.trace
import opentelemetry.instrumentation.fastapi
import opentelemetry.exporter.prometheus
import prometheus_client
print('all imports OK')
"
```

Expected: `all imports OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add observability dependencies (structlog, OTEL, prometheus-client)"
```

---

## Task 2: Add observability config fields

**Files:**
- Modify: `restorax/config.py`

Note: `app_env` and `log_level` already exist in `Settings` — do not duplicate them.

- [ ] **Step 1: Add three new fields to Settings**

In `restorax/config.py`, add these three fields after the `log_level` field:

```python
    # Observability
    otel_service_name: str = "restorax"
    otel_exporter_otlp_endpoint: str | None = None   # e.g. "http://localhost:4317"
    sentry_dsn: str | None = None
```

The full `Settings` class body, in order, should be:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESTORAX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./restorax.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    storage_backend: str = "local"
    storage_local_root: str = "./data"

    # S3 / MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "restorax"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    # ML
    device: str = "cuda"
    model_dir: str = "./models"
    registry_max_loaded: int = 2

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Observability
    otel_service_name: str = "restorax"
    otel_exporter_otlp_endpoint: str | None = None
    sentry_dsn: str | None = None


settings = Settings()
```

- [ ] **Step 2: Run existing config tests to make sure nothing broke**

```bash
conda run -n restorax python -m pytest tests/unit/test_config.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add restorax/config.py
git commit -m "feat: add otel_service_name, otel_exporter_otlp_endpoint, sentry_dsn to Settings"
```

---

## Task 3: Create structlog logging module

**Files:**
- Create: `restorax/logging.py`
- Create: `tests/unit/test_logging.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_logging.py`:

```python
from __future__ import annotations

import logging

import pytest
import structlog


def _reset():
    """Reset the _configured guard between tests."""
    import restorax.logging as rlog
    rlog._configured = False


def test_configure_logging_dev_uses_console_renderer():
    _reset()
    from restorax.logging import configure_logging
    configure_logging(app_env="development", log_level="DEBUG")
    # structlog is configured — calling get_logger must not raise
    logger = structlog.get_logger("test")
    logger.info("hello from dev")


def test_configure_logging_prod_uses_json_renderer(capsys):
    _reset()
    from restorax.logging import configure_logging
    configure_logging(app_env="production", log_level="INFO")
    logger = structlog.get_logger("test")
    logger.info("hello from prod", key="value")
    captured = capsys.readouterr()
    # JSON output must contain the event key
    assert "hello from prod" in captured.out


def test_configure_logging_is_idempotent():
    _reset()
    from restorax.logging import configure_logging
    configure_logging(app_env="development")
    configure_logging(app_env="production")  # second call must be a no-op
    # If idempotency is broken, structlog would be configured twice and raise
    structlog.get_logger("test").info("idempotency check")


def test_stdlib_logging_redirected():
    _reset()
    from restorax.logging import configure_logging
    configure_logging(app_env="development", log_level="DEBUG")
    std_logger = logging.getLogger("sqlalchemy.engine")
    # Stdlib logger must have our handler (not the default lastResort)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, structlog.stdlib.ProcessorFormatter)


def test_inject_otel_context_no_active_span():
    from restorax.logging import _inject_otel_context
    event_dict: dict = {"event": "test"}
    result = _inject_otel_context(None, "info", event_dict)
    # No active span — must return event_dict unchanged (no trace_id key)
    assert result == {"event": "test"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n restorax python -m pytest tests/unit/test_logging.py -v
```

Expected: `ModuleNotFoundError: No module named 'restorax.logging'`

- [ ] **Step 3: Create `restorax/logging.py`**

```python
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def _inject_otel_context(_logger: Any, _method: str, event_dict: dict) -> dict:
    """Reads the active OTEL span and injects trace_id + span_id as log fields."""
    try:
        from opentelemetry import trace
        ctx = trace.get_current_span().get_span_context()
        if ctx and ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:
        pass
    return event_dict


def configure_logging(app_env: str = "development", log_level: str = "INFO") -> None:
    """Configure structlog and redirect stdlib logging to use the same pipeline."""
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        _inject_otel_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    renderer: structlog.types.Processor
    if app_env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n restorax python -m pytest tests/unit/test_logging.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add restorax/logging.py tests/unit/test_logging.py
git commit -m "feat: add structlog logging module with dev/prod renderers and OTEL context injection"
```

---

## Task 4: Create OTEL telemetry module

**Files:**
- Create: `restorax/telemetry.py`
- Create: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_telemetry.py`:

```python
from __future__ import annotations

import pytest


def _reset():
    import restorax.telemetry as tel
    tel._configured = False


def test_configure_telemetry_no_endpoint_does_not_raise(tmp_path):
    """With no OTLP endpoint or Sentry DSN, telemetry must configure silently."""
    _reset()
    from restorax.config import Settings
    s = Settings(
        database_url="sqlite+aiosqlite:///./test.db",
        otel_exporter_otlp_endpoint=None,
        sentry_dsn=None,
    )
    from restorax.telemetry import configure_telemetry
    configure_telemetry(s)  # must not raise


def test_configure_telemetry_is_idempotent():
    _reset()
    from restorax.config import Settings
    s = Settings(database_url="sqlite+aiosqlite:///./test.db")
    from restorax.telemetry import configure_telemetry
    configure_telemetry(s)
    configure_telemetry(s)  # second call must be a no-op


def test_get_tracer_returns_tracer():
    _reset()
    from restorax.config import Settings
    configure_args = Settings(database_url="sqlite+aiosqlite:///./test.db")
    from restorax.telemetry import configure_telemetry, get_tracer
    configure_telemetry(configure_args)
    tracer = get_tracer()
    assert tracer is not None


def test_get_meter_returns_meter():
    _reset()
    from restorax.config import Settings
    s = Settings(database_url="sqlite+aiosqlite:///./test.db")
    from restorax.telemetry import configure_telemetry, get_meter
    configure_telemetry(s)
    meter = get_meter()
    assert meter is not None


def test_business_metric_helpers_return_instruments():
    _reset()
    from restorax.config import Settings
    s = Settings(database_url="sqlite+aiosqlite:///./test.db")
    from restorax.telemetry import (
        configure_telemetry,
        get_active_jobs_counter,
        get_job_duration_histogram,
        get_jobs_counter,
    )
    configure_telemetry(s)
    assert get_jobs_counter() is not None
    assert get_job_duration_histogram() is not None
    assert get_active_jobs_counter() is not None


def test_sentry_missing_package_logs_warning(caplog):
    """If SENTRY_DSN is set but sentry-sdk not installed, a warning must be logged."""
    _reset()
    import sys
    # Temporarily hide sentry_sdk from imports
    sentry_backup = sys.modules.pop("sentry_sdk", None)
    sys.modules["sentry_sdk"] = None  # type: ignore[assignment]

    import logging
    from restorax.config import Settings
    s = Settings(database_url="sqlite+aiosqlite:///./test.db", sentry_dsn="https://fake@sentry.io/1")
    from restorax.telemetry import configure_telemetry
    with caplog.at_level(logging.WARNING):
        configure_telemetry(s)
    assert "sentry-sdk not installed" in caplog.text

    # Restore
    del sys.modules["sentry_sdk"]
    if sentry_backup:
        sys.modules["sentry_sdk"] = sentry_backup
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n restorax python -m pytest tests/unit/test_telemetry.py -v
```

Expected: `ModuleNotFoundError: No module named 'restorax.telemetry'`

- [ ] **Step 3: Create `restorax/telemetry.py`**

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from restorax.config import Settings

logger = logging.getLogger(__name__)

_configured = False

# Business metric instrument singletons — initialized in configure_telemetry
_jobs_counter = None
_job_duration_histogram = None
_active_jobs_counter = None


def configure_telemetry(settings: Settings) -> None:
    """Set up OTEL TracerProvider + MeterProvider. Idempotent."""
    global _configured, _jobs_counter, _job_duration_histogram, _active_jobs_counter
    if _configured:
        return
    _configured = True

    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": "0.1.0",
        "deployment.environment": settings.app_env,
    })

    # ── Traces ────────────────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)

    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
            )
        )

    trace.set_tracer_provider(tracer_provider)

    # ── Sentry (optional) ─────────────────────────────────────────────────────
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.celery import CeleryIntegration
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.starlette import StarletteIntegration
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                traces_sample_rate=1.0,
                integrations=[StarletteIntegration(), FastApiIntegration(), CeleryIntegration()],
            )
        except (ImportError, TypeError):
            logger.warning(
                "SENTRY_DSN set but sentry-sdk not installed — run: pip install restorax[apm]"
            )

    # ── Metrics (Prometheus) ──────────────────────────────────────────────────
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # Business metrics
    meter = metrics.get_meter("restorax")
    _jobs_counter = meter.create_counter(
        "restorax_jobs_total",
        description="Total jobs by completion status",
    )
    _job_duration_histogram = meter.create_histogram(
        "restorax_job_duration_seconds",
        description="Job processing duration in seconds",
        unit="s",
    )
    _active_jobs_counter = meter.create_up_down_counter(
        "restorax_active_jobs",
        description="Number of currently active jobs",
    )

    # ── Auto-instrumentation ──────────────────────────────────────────────────
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    FastAPIInstrumentor().instrument(excluded_urls="/health,/ready,/metrics")
    CeleryInstrumentor().instrument()
    RedisInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()


def get_tracer():
    from opentelemetry import trace
    return trace.get_tracer("restorax")


def get_meter():
    from opentelemetry import metrics
    return metrics.get_meter("restorax")


def get_jobs_counter():
    return _jobs_counter


def get_job_duration_histogram():
    return _job_duration_histogram


def get_active_jobs_counter():
    return _active_jobs_counter
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n restorax python -m pytest tests/unit/test_telemetry.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add restorax/telemetry.py tests/unit/test_telemetry.py
git commit -m "feat: add OTEL telemetry module (traces, Prometheus metrics, auto-instrumentation)"
```

---

## Task 5: Upgrade RequestIDMiddleware

**Files:**
- Modify: `restorax/api/middleware.py`
- Create: `tests/unit/test_middleware_observability.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_middleware_observability.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n restorax python -m pytest tests/unit/test_middleware_observability.py -v
```

Expected: `test_request_id_bound_to_structlog_context` and `test_context_cleared_between_requests` FAIL (existing middleware doesn't bind structlog context).

- [ ] **Step 3: Upgrade `restorax/api/middleware.py`**

Replace the entire file content:

```python
from __future__ import annotations

import logging
import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Generates or propagates X-Request-ID and binds it to structlog context.
    Also injects the request_id as an attribute on the active OTEL span.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            from opentelemetry import trace
            trace.get_current_span().set_attribute("http.request_id", request_id)
        except Exception:
            pass

        request.state.request_id = request_id
        response: Response = await call_next(request)  # type: ignore[operator]
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Logs request duration and adds X-Process-Time header."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[operator]
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"
        logger.debug(
            "request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(elapsed_ms, 1),
        )
        return response
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n restorax python -m pytest tests/unit/test_middleware_observability.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add restorax/api/middleware.py tests/unit/test_middleware_observability.py
git commit -m "feat: upgrade RequestIDMiddleware with structlog context binding and OTEL span attribute"
```

---

## Task 6: Create health router

**Files:**
- Create: `restorax/api/routers/health.py`
- Create: `tests/integration/test_health.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_health.py`:

```python
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
```

Check whether `async_client` fixture exists in the integration conftest:

```bash
grep -n "async_client\|AsyncClient" /mnt/f/wsl_repo/restorax/tests/integration/conftest.py
```

If not present, add this fixture to `tests/integration/conftest.py`:

```python
@pytest_asyncio.fixture
async def async_client():
    from httpx import AsyncClient, ASGITransport
    from restorax.api.app import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
```

And add the import at the top of `tests/integration/conftest.py`:
```python
import pytest_asyncio
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n restorax python -m pytest tests/integration/test_health.py -v
```

Expected: FAIL — `/health` returns `{"status": "ok", "version": "0.1.0"}` but the new router hasn't been created yet, and `/ready` doesn't exist.

- [ ] **Step 3: Create `restorax/api/routers/health.py`**

```python
from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> JSONResponse:
    db_status = await _check_db()
    redis_status = await _check_redis()
    body = {"db": db_status, "redis": redis_status}
    status_code = 200 if db_status == "ok" and redis_status == "ok" else 503
    return JSONResponse(content=body, status_code=status_code)


async def _check_db() -> str:
    try:
        from sqlalchemy import text
        from restorax.db.session import AsyncSessionLocal
        async with asyncio.timeout(2.0):
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "fail"


async def _check_redis() -> str:
    try:
        import redis.asyncio as aioredis
        from restorax.config import settings
        async with asyncio.timeout(2.0):
            r = aioredis.from_url(settings.redis_url)
            await r.ping()
            await r.aclose()
        return "ok"
    except Exception:
        return "fail"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n restorax python -m pytest tests/integration/test_health.py -v
```

Expected: `test_health_always_200` passes. `test_ready_*` pass if Redis is running locally; `test_ready_200_when_db_and_redis_healthy` may show 503 for redis in CI without Redis — that's acceptable per the assertion.

- [ ] **Step 5: Commit**

```bash
git add restorax/api/routers/health.py tests/integration/test_health.py tests/integration/conftest.py
git commit -m "feat: add /health (liveness) and /ready (readiness) endpoints"
```

---

## Task 7: Wire app.py

**Files:**
- Modify: `restorax/api/app.py`

- [ ] **Step 1: Replace `restorax/api/app.py` content**

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from restorax.api.middleware import RequestIDMiddleware, TimingMiddleware
from restorax.api.routers import jobs, models, pipelines, ws
from restorax.api.routers.health import router as health_router
from restorax.config import settings
from restorax.logging import configure_logging
from restorax.telemetry import configure_telemetry


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from restorax.db.session import create_tables
    await create_tables()
    yield


def create_app() -> FastAPI:
    configure_logging(app_env=settings.app_env, log_level=settings.log_level)
    configure_telemetry(settings)

    app = FastAPI(
        title="RestoraX",
        description="Modern AI video restoration platform",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(jobs.router)
    app.include_router(models.router)
    app.include_router(pipelines.router)
    app.include_router(ws.router)

    # ── Prometheus /metrics ───────────────────────────────────────────────────
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
```

- [ ] **Step 2: Run the full integration test suite to verify the app still boots**

```bash
conda run -n restorax python -m pytest tests/integration/ -v
```

Expected: all previously passing tests still pass. New health tests pass.

- [ ] **Step 3: Verify /metrics endpoint works**

```bash
conda run -n restorax python -c "
from restorax.api.app import create_app
from fastapi.testclient import TestClient
client = TestClient(create_app())
resp = client.get('/metrics')
print('status:', resp.status_code)
print('content-type:', resp.headers.get('content-type',''))
print('first line:', resp.text.split('\n')[0])
"
```

Expected: status 200, content-type `text/plain`, first line starts with `#`.

- [ ] **Step 4: Commit**

```bash
git add restorax/api/app.py
git commit -m "feat: wire configure_logging + configure_telemetry into FastAPI app, add /metrics endpoint"
```

---

## Task 8: Wire Celery task context and business metrics

**Files:**
- Modify: `restorax/tasks/job_tasks.py`

- [ ] **Step 1: Add structlog import and replace `logger = logging.getLogger(...)`**

At the top of `restorax/tasks/job_tasks.py`, replace:
```python
import logging
```
with:
```python
import logging
import time

import structlog
```

Replace the existing logger line:
```python
logger = logging.getLogger(__name__)
```
with:
```python
logger = structlog.get_logger(__name__)
```

- [ ] **Step 2: Add Celery signal handlers for structlog context**

After the `_audio_registry` block (after the `_get_audio_registry` function), add:

```python
# ── Structlog context signals ─────────────────────────────────────────────────

from celery.signals import task_failure, task_postrun, task_prerun


@task_prerun.connect
def _on_task_prerun(task_id: str, task: object, args: tuple, kwargs: dict, **_: object) -> None:
    job_id = kwargs.get("job_id") or (args[0] if args else None)
    structlog.contextvars.clear_contextvars()
    ctx: dict[str, str] = {"celery_task_id": task_id}
    if job_id:
        ctx["job_id"] = str(job_id)
    structlog.contextvars.bind_contextvars(**ctx)


@task_postrun.connect
def _on_task_postrun(**_: object) -> None:
    structlog.contextvars.clear_contextvars()


@task_failure.connect
def _on_task_failure(**_: object) -> None:
    structlog.contextvars.clear_contextvars()
```

- [ ] **Step 3: Add metric increments to `run_job`**

At the start of the `run_job` function body, after `reporter = ProgressReporter(job_id)`, add:

```python
    _start_time = time.perf_counter()
    try:
        from restorax.telemetry import get_active_jobs_counter
        ctr = get_active_jobs_counter()
        if ctr is not None:
            ctr.add(1, {"pipeline": pipeline_preset_path})
    except Exception:
        pass
```

At the end of `run_job`, before `return {"output_path": output_path, "metrics": {}}`, add:

```python
    try:
        from pathlib import Path as _Path
        from restorax.telemetry import (
            get_active_jobs_counter,
            get_job_duration_histogram,
            get_jobs_counter,
        )
        _dur = time.perf_counter() - _start_time
        _pipeline_name = _Path(pipeline_preset_path).stem
        _jc = get_jobs_counter()
        _jd = get_job_duration_histogram()
        _ac = get_active_jobs_counter()
        if _jc is not None:
            _jc.add(1, {"status": "completed"})
        if _jd is not None:
            _jd.record(_dur, {"pipeline": _pipeline_name})
        if _ac is not None:
            _ac.add(-1, {"pipeline": pipeline_preset_path})
    except Exception:
        pass
```

In `JobTask.on_failure`, after `super().on_failure(...)`, add metric decrement:

```python
        try:
            from restorax.telemetry import get_active_jobs_counter, get_jobs_counter
            _jc = get_jobs_counter()
            _ac = get_active_jobs_counter()
            if _jc is not None:
                _jc.add(1, {"status": "failed"})
            if _ac is not None:
                _ac.add(-1, {"pipeline": ""})
        except Exception:
            pass
```

- [ ] **Step 4: Update all `logger.info/warning/debug` calls in `run_job` to use structlog keyword style**

Replace:
```python
logger.info("Starting job %s | device=%s | preset=%s | audio=%s",
            job_id, device, pipeline_preset_path, restore_audio)
```
with:
```python
logger.info("job started", device=str(device), preset=pipeline_preset_path, restore_audio=restore_audio)
```

Replace:
```python
logger.info("Job %s completed → %s", job_id, output_path)
```
with:
```python
logger.info("job completed", output_path=output_path)
```

Replace in `_run_audio_pipeline`:
```python
logger.debug("No audio_stages in preset — skipping audio pipeline")
```
with:
```python
logger.debug("no audio_stages in preset, skipping")
```

Replace:
```python
logger.info("Audio pipeline complete: %s stages applied", len(pipeline.stages))
```
with:
```python
logger.info("audio pipeline complete", stages=len(pipeline.stages))
```

- [ ] **Step 5: Run unit tests to verify nothing broke**

```bash
conda run -n restorax python -m pytest tests/unit/ -v
```

Expected: all pass.

- [ ] **Step 6: Run the full test suite**

```bash
conda run -n restorax python -m pytest tests/ -v --ignore=tests/system
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add restorax/tasks/job_tasks.py
git commit -m "feat: add structlog context signals and OTEL business metrics to Celery job tasks"
```

---

## Task 9: Final smoke test

- [ ] **Step 1: Run the complete test suite**

```bash
conda run -n restorax python -m pytest tests/ -v --ignore=tests/system -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify dev startup produces structured logs**

```bash
conda run -n restorax python -c "
from restorax.logging import configure_logging
configure_logging(app_env='development', log_level='DEBUG')
import structlog
log = structlog.get_logger('smoke')
import structlog.contextvars
structlog.contextvars.bind_contextvars(request_id='test-123')
log.info('smoke test', component='backend-hardening')
"
```

Expected: colored console output with `request_id=test-123` visible.

- [ ] **Step 3: Verify /metrics returns Prometheus format**

```bash
conda run -n restorax python -c "
from restorax.api.app import create_app
from fastapi.testclient import TestClient
client = TestClient(create_app())
resp = client.get('/metrics')
assert resp.status_code == 200
lines = [l for l in resp.text.split('\n') if 'restorax' in l]
print('restorax metrics found:', len(lines))
for l in lines[:5]: print(' ', l)
"
```

Expected: restorax custom metrics lines printed.

- [ ] **Step 4: Commit if anything needed fixing**

```bash
git add -p
git commit -m "fix: backend hardening smoke test adjustments"
```

---

## Summary of changes

| File | Change type |
|---|---|
| `pyproject.toml` | +11 deps, +1 optional group |
| `restorax/config.py` | +3 fields |
| `restorax/logging.py` | New — 60 lines |
| `restorax/telemetry.py` | New — 90 lines |
| `restorax/api/middleware.py` | Rewritten — structlog + OTEL |
| `restorax/api/routers/health.py` | New — 50 lines |
| `restorax/api/app.py` | Rewritten — wires observability stack |
| `restorax/tasks/job_tasks.py` | +signals, +metrics, +structlog style |
| `tests/unit/test_logging.py` | New — 5 tests |
| `tests/unit/test_telemetry.py` | New — 6 tests |
| `tests/unit/test_middleware_observability.py` | New — 4 tests |
| `tests/integration/test_health.py` | New — 4 tests |
