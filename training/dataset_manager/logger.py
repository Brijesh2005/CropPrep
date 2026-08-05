"""Structured logging for the Dataset Manager.

The module exposes two entry points:

* :func:`setup_logging` — configure the root logger once with a rotating file
  handler and (optionally) a console handler.
* :func:`get_logger` — obtain a namespaced child logger for a module.

The file handler emits JSON lines (one record per line) so logs are directly
consumable by aggregators (ELK, Loki, ...). The console handler emits a
compact human readable format. Both support all standard log levels and the
``extra`` keyword, which is embedded into the JSON payload.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

from shared.logging.formatters import CompactFormatter, JsonFormatter, RESERVED as _RESERVED

_configured = False


def setup_logging(
    *,
    level: str | int = logging.INFO,
    log_dir: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    json_format: bool = True,
    console: bool = True,
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """Configure the Dataset Manager logger hierarchy.

    Args:
        level: Log level name or numeric level (e.g. ``"DEBUG"``).
        log_dir: Directory for rotating log files. When omitted only the
            console handler is installed (no file handler).
        max_bytes: Size cap of each log file before rotation.
        backup_count: Number of rotated files to retain.
        json_format: Use the JSON file formatter (console stays compact).
        console: Also install a console handler.
        logger: Root logger to configure. Defaults to the package logger.

    Returns:
        The configured logger.
    """
    global _configured

    root = logger or logging.getLogger("cropfusion.dataset_manager")
    numeric = logging.getLevelName(level) if isinstance(level, str) else level
    root.setLevel(numeric)

    # Remove any handlers previously installed by this module so repeated
    # calls are idempotent.
    for handler in list(root.handlers):
        if getattr(handler, "_dm_owned", False):
            root.removeHandler(handler)

    if console:
        # Logs go to stderr so stdout stays clean for data output (the CLI
        # prints JSON reports to stdout).
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(CompactFormatter())
        console_handler.setLevel(numeric)
        setattr(console_handler, "_dm_owned", True)
        root.addHandler(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "dataset_manager.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter() if json_format else CompactFormatter())
        file_handler.setLevel(numeric)
        setattr(file_handler, "_dm_owned", True)
        root.addHandler(file_handler)

    root.propagate = False
    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for ``name`` (e.g. ``"downloader"``).

    Loggers are created under the package hierarchy, so ``setup_logging``
    configuration applies to them.
    """
    if name.startswith("cropfusion."):
        return logging.getLogger(name)
    return logging.getLogger(f"cropfusion.dataset_manager.{name}")


def is_configured() -> bool:
    """True when :func:`setup_logging` has been called."""
    return _configured


def log_dict(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured log entry: ``logger.log(level, event, extra=fields)``."""
    logger.log(level, event, extra=_safe_extra(fields))


def _safe_extra(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Strip keys that collide with reserved LogRecord attributes."""
    return {k: v for k, v in fields.items() if k not in _RESERVED}
