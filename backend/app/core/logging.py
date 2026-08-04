"""Structured logging (loguru).

* JSON output for machine parsing (or readable text in development).
* Optional rotating file sink.
* Correlation IDs propagated through ``contextvars``.
* Performance logging helpers.
"""

from __future__ import annotations

import contextvars
import json
import sys
import time
from typing import Any

from loguru import logger

from .config import LogSettings

#: Correlation (request) ID shared across the request lifecycle.
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(request_id: str | None) -> None:
    _correlation_id.set(request_id)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


class _JsonFormatter:
    """Serialize loguru records as JSON lines.

    loguru calls ``format_map(record)`` on the returned string, so JSON braces
    must be escaped (``{{``/``}}``) for the output to be the literal JSON.
    """

    def __call__(self, record: dict[str, Any]) -> str:
        payload = {
            "ts": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "logger": record["name"],
            "correlation_id": get_correlation_id(),
            "context": record.get("extra", {}),
        }
        exception = record.get("exception")
        if exception is not None:
            payload["exception"] = str(exception)
        line = json.dumps(payload, default=str) + "\n"
        return line.replace("{", "{{").replace("}", "}}")


def setup_logging(settings: LogSettings) -> None:
    """Configure the root loguru logger from ``settings``."""
    logger.remove()
    formatter = _JsonFormatter() if settings.json_logs else (
        "{time} | {level: <8} | {message}"
    )
    logger.add(
        sys.stdout,
        level=settings.level.upper(),
        format=formatter,
        enqueue=True,
    )
    if settings.log_dir:
        from pathlib import Path

        Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            str(Path(settings.log_dir) / "backend.log"),
            level=settings.level.upper(),
            format=formatter,
            rotation=f"{settings.max_bytes} bytes",
            retention=settings.backup_count,
            enqueue=True,
        )


# Re-export the configured logger.
get_logger = lambda name=None: logger.bind(module=name)  # noqa: E731


class PerformanceTimer:
    """Record a timed block and emit a performance log line."""

    def __init__(self, operation: str, **tags: Any) -> None:
        self.operation = operation
        self.tags = tags
        self._start = time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._start
        logger.bind(
            operation=self.operation,
            duration_ms=round(elapsed * 1000, 3),
            **self.tags,
        ).info("performance")
        return elapsed

    def __enter__(self) -> "PerformanceTimer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
