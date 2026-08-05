"""Log formatters: JSON, compact console and colored console."""

from __future__ import annotations

import json
import logging
from typing import Any

#: Reserved LogRecord attributes that must not be re-emitted verbatim.
RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)

#: ANSI colours for the colored console formatter.
_COLORS = {
    "DEBUG": "\x1b[36m",
    "INFO": "\x1b[32m",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
    "CRITICAL": "\x1b[1;31m",
}
_RESET = "\x1b[0m"


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        # Fold in any custom attributes passed via `extra=`.
        for key, value in record.__dict__.items():
            if key not in RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


class CompactFormatter(logging.Formatter):
    """Single line, human readable console formatter."""

    def format(self, record: logging.LogRecord) -> str:
        base = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
        formatter = logging.Formatter(base, datefmt="%H:%M:%S")
        text = formatter.format(record)
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


class ColoredFormatter(logging.Formatter):
    """Console formatter with ANSI colour per log level."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        base = f"%(asctime)s {color}%(levelname)-8s{_RESET} %(name)s: %(message)s"
        formatter = logging.Formatter(base, datefmt="%H:%M:%S")
        text = formatter.format(record)
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text
