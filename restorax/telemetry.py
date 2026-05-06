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
