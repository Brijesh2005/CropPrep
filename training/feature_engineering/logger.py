"""Structured logging for the feature-engineering package.

Loggers live under ``cropfusion.feature_engineering.*`` and reuse the shared
formatters so every platform logger stays JSON-capable and consistent.
"""

from __future__ import annotations

import logging

from shared.logging.formatters import CompactFormatter, JsonFormatter

ROOT_NAME = "cropfusion.feature_engineering"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger("tabular")``."""
    if name.startswith(ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def setup_logging(
    *,
    level: str | int = logging.INFO,
    log_dir: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    json_format: bool = True,
    console: bool = True,
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """Configure the feature-engineering logger hierarchy."""
    from logging.handlers import RotatingFileHandler
    import sys

    root = logger or get_logger("")
    numeric = logging.getLevelName(level) if isinstance(level, str) else level
    root.setLevel(numeric)

    for handler in list(root.handlers):
        if getattr(handler, "_fe_owned", False):
            root.removeHandler(handler)

    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(CompactFormatter())
        handler.setLevel(numeric)
        setattr(handler, "_fe_owned", True)
        root.addHandler(handler)

    if log_dir:
        from pathlib import Path

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            Path(log_dir) / "feature_engineering.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter() if json_format else CompactFormatter())
        handler.setLevel(numeric)
        setattr(handler, "_fe_owned", True)
        root.addHandler(handler)

    root.propagate = False
    return root
