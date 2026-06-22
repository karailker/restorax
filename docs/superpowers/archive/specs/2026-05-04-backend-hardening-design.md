# RestoraX Backend Hardening — Design Spec

**Date:** 2026-05-04
**Track:** 1 of 5 (Backend Hardening)
**Status:** Approved

## Goal

Bring the RestoraX backend to production-grade observability standards while keeping
zero extra containers in development. Structured logs, distributed traces, Prometheus
metrics, and health probes all activate from environment variables — no code changes
needed to switch between dev and prod.

## Approach

Option C (approved): OTEL SDK with environment-driven multi-export. structlog as the
log backend wired into OTEL. Prometheus `/metrics` baked into the FastAPI app.
Sentry optional via `SENTRY_DSN` env var.

## Architecture

### New files

| File | Purpose |
|---|---|
| `restorax/logging.py` | structlog configuration — pretty in dev, JSON in prod |
| `restorax/telemetry.py` | OTEL TracerProvider + MeterProvider setup, Sentry wiring |
| `restorax/api/routers/health.py` | `/health` (liveness) and `/ready` (readiness) endpoints |

### Modified files

| File | Change |
|---|---|
| `restorax/config.py` | Add `RESTORAX_ENV`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `LOG_LEVEL`, `SENTRY_DSN` settings |
| `restorax/api/app.py` | Call `configure_logging()` + `configure_telemetry()` at startup; mount health router |
| `restorax/api/middleware.py` | Add `RequestIDMiddleware` — generates/propagates `X-Request-ID` |
| `restorax/tasks/job_tasks.py` | Bind `job_id` + `celery_task_id` to structlog context per task |
| `pyproject.toml` | Add observability dependencies; add `[apm]` optional group for sentry-sdk |

### Boot sequence

```
configure_logging(settings)
  → configure_telemetry(settings)
    → FastAPI app created
      → RequestIDMiddleware attached
        → routers mounted (including /health, /ready)
```

Both `configure_*` functions are idempotent and safe to call multiple times (guarded
by a module-level `_configured` flag).

## Section 1 — Structured Logging (`restorax/logging.py`)

### Processor chain

```
add_log_level
→ add_timestamp (ISO 8601)
→ add_logger_name
→ contextvars_merge          # picks up all bound context (request_id, job_id, etc.)
→ inject_otel_context        # custom processor: reads opentelemetry.trace.get_current_span(),
                             #   injects trace_id + span_id as log fields (no-op if no active span)
→ [dev]  ConsoleRenderer     # colored, human-readable
→ [prod] JSONRenderer        # one JSON object per line
```

### Environment switching

`configure_logging(settings)` reads:
- `settings.RESTORAX_ENV` — `"development"` → ConsoleRenderer, `"production"` → JSONRenderer
- `settings.LOG_LEVEL` — passed to both structlog and the stdlib `logging` root logger

### stdlib redirect

`logging.basicConfig` is called with a `structlog.stdlib.ProcessorFormatter` handler
so that SQLAlchemy, Celery, uvicorn, and all third-party loggers emit in the same
format as structlog output.

### Context binding

Any code can call:
```python
import structlog
structlog.contextvars.bind_contextvars(request_id="...", job_id="...")
```
Every subsequent log call in that context automatically includes those fields.
Context is cleared between requests by the `RequestIDMiddleware` using
`structlog.contextvars.clear_contextvars()`.

## Section 2 — OTEL Telemetry (`restorax/telemetry.py`)

### Traces

Auto-instrumented via OTEL packages:
- `FastAPIInstrumentor` — spans for every HTTP request
- `CeleryInstrumentor` — spans for every task publish + execution
- `SQLAlchemyInstrumentor` — spans for every DB query
- `RedisInstrumentor` — spans for every Redis command

Manual spans added in `restorax/core/pipeline.py` for per-restorer timing:
```python
with tracer.start_as_current_span("restorer.run", attributes={"restorer": name}):
    ...
```

### Metrics

`PrometheusMetricReader` attached to the OTEL `MeterProvider`. Three custom
business metrics defined in `restorax/telemetry.py` and incremented from
`job_tasks.py`:

```
restorax_jobs_total{status="completed|failed|cancelled"}   # Counter
restorax_job_duration_seconds{pipeline="<name>"}           # Histogram (buckets: 1s–600s)
restorax_active_jobs                                       # UpDownCounter
```

### Export — environment-driven

| Env var set | Trace export | Metrics |
|---|---|---|
| Neither | Stdout (dev) | `/metrics` on app |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP gRPC to endpoint | `/metrics` on app |
| `SENTRY_DSN` | Sentry (via SentrySpanProcessor) + OTLP if endpoint also set | `/metrics` on app |

`/metrics` is always available regardless of export config.

### Sentry integration

```python
if settings.SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=1.0,
        instrument_fastapi=True,
    )
    # Wire Sentry as an OTEL span processor so it receives OTEL traces
    from opentelemetry.sdk.trace import TracerProvider
    provider.add_span_processor(SentrySpanProcessor())
```

`sentry-sdk` is only imported when `SENTRY_DSN` is present. The import is guarded
with a clear `ImportError` message directing to `pip install restorax[apm]`.

## Section 3 — Health Probes (`restorax/api/routers/health.py`)

### `GET /health` — liveness

Returns `200 {"status": "ok"}` unconditionally. Purpose: tell the orchestrator the
process is alive. Never performs I/O.

### `GET /ready` — readiness

Runs two checks concurrently (asyncio.gather):
1. `SELECT 1` via the SQLAlchemy async session
2. `PING` via the aioredis client

Response:
```json
{"db": "ok", "redis": "ok"}   → 200
{"db": "ok", "redis": "fail"} → 503
{"db": "fail", "redis": "ok"} → 503
```

Timeout: each check has a 2-second timeout; a timeout counts as `"fail"`.

Both endpoints are unauthenticated and excluded from OTEL request span sampling
(added to `exclude_spans` list) to avoid noise in traces.

## Section 4 — Request ID Propagation (`RequestIDMiddleware`)

Added to `restorax/api/middleware.py`.

**Per-request lifecycle:**
1. Read `X-Request-ID` from incoming headers; generate `uuid4()` if absent
2. `structlog.contextvars.clear_contextvars()` — prevent context leak between requests
3. `structlog.contextvars.bind_contextvars(request_id=request_id)`
4. `trace.get_current_span().set_attribute("http.request_id", request_id)`
5. Process request
6. Add `X-Request-ID: <id>` to response headers

**Celery task context:**
- `task_prerun` signal: `bind_contextvars(job_id=..., celery_task_id=task.request.id)`
- `task_postrun` / `task_failure` signals: `clear_contextvars()`

Every log line from a Celery worker automatically includes the job it's processing.

## Section 5 — Config additions (`restorax/config.py`)

```python
RESTORAX_ENV: str = "development"           # "development" | "production"
LOG_LEVEL: str = "INFO"
OTEL_SERVICE_NAME: str = "restorax"
OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None   # e.g. "http://localhost:4317"
SENTRY_DSN: str | None = None
```

All fields read from environment variables via `pydantic-settings` (already in use).

## Section 6 — Dependencies

### Core (added to `[project.dependencies]`)

```toml
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

### Optional APM (`[project.optional-dependencies]`)

```toml
[apm]
"sentry-sdk[fastapi,celery]>=2.0.0",
```

Install with: `pip install restorax[apm]`

## Testing

- Unit tests for `configure_logging()` — verify processor chain in dev vs prod mode
- Unit tests for `configure_telemetry()` — verify no-op when no endpoint set
- Unit test for `RequestIDMiddleware` — verify header generation, propagation, response
- Integration test for `/health` — always 200
- Integration test for `/ready` — 200 with healthy DB+Redis; 503 with mocked failures
- Unit tests for custom metrics — verify counter/histogram increments in job lifecycle

All new tests go in `tests/unit/` and `tests/integration/`. No GPU required.

## Out of Scope

- Grafana dashboards or alert rules (bring-your-own-backend)
- Log aggregation configuration (Loki, Fluentd, etc.)
- Sentry project setup / DSN provisioning
- Changes to the Celery beat scheduler or task routing
- Any changes to AI pipeline, frontend, or documentation
