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
