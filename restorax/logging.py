from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def _inject_otel_context(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Reads the active OTEL span and injects trace_id + span_id as log fields."""
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx and ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:  # noqa: S110 - logging processor must never raise or log recursively
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
        _inject_otel_context,  # type: ignore[list-item]
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
