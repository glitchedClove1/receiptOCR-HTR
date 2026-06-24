"""Structured logging.

Emits one JSON object per log line so logs are machine-parseable in production
(searchable by field, ingestible by log aggregators). A per-request
correlation id is carried via a ContextVar so every log line produced while
handling a request can be tied back to that request without threading the id
through every function call.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

# Set per request by the API middleware; included in every log record.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        # Attach any structured extras passed via logger.info(..., extra={...}).
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a logger adapter that accepts a `fields` dict for structured logs.

    Usage:
        log = get_logger(__name__)
        log.info("processed", fields={"regions": 12, "cache_hit": False})
    """

    base = logging.getLogger(name)

    class _Adapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            fields = kwargs.pop("fields", None)
            if fields:
                extra = kwargs.setdefault("extra", {})
                extra["extra_fields"] = fields
            return msg, kwargs

    return _Adapter(base, {})
