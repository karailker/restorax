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
