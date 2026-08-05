"""Structured logging setup for the CropFusion platforms.

Provides a single :func:`setup_logging` entry point that supports JSON file
logging, rotating file handlers, console handlers (compact or colored) and
named profiles for the Training platform, the Application platform and audit
trails.  Logger names follow the ``cropfusion.<namespace>`` hierarchy so the
existing platform loggers (``cropfusion.dataset_manager``,
``cropfusion.spatial_alignment``, ...) keep working.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from ..exceptions import LoggingConfigurationError
from .formatters import ColoredFormatter, CompactFormatter, JsonFormatter

#: Valid setup profiles.
PROFILES = frozenset({"default", "training", "application", "audit"})

#: Valid output formats.
FORMATS = frozenset({"json", "compact", "colored"})

#: Default log file names per profile.
PROFILE_FILE_NAMES = {
    "training": "training.log",
    "application": "application.log",
    "audit": "audit.log",
    "default": "cropfusion.log",
}

_configured = False


def _build_formatter(output: str) -> logging.Formatter:
    if output == "json":
        return JsonFormatter()
    if output == "colored":
        return ColoredFormatter()
    return CompactFormatter()


def _mark_owned(handler: logging.Handler, marker: str) -> None:
    setattr(handler, marker, True)


def setup_logging(
    *,
    level: str | int = logging.INFO,
    log_dir: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    output: str = "json",
    console: bool = True,
    console_output: str = "compact",
    profile: str = "default",
    logger_name: str | None = None,
    file_name: str | None = None,
    rotate: bool = True,
    marker: str = "_cf_owned",
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """Configure a logger hierarchy.

    Args:
        level: Log level name or numeric level.
        log_dir: Optional directory for file logging. When omitted only the
            console handler is installed.
        max_bytes: Size cap of each log file before rotation.
        backup_count: Rotated files retained.
        output: File output format — ``"json"``, ``"compact"`` or ``"colored"``.
        console: Install a stderr console handler.
        console_output: Console format — ``"compact"`` or ``"colored"``.
        profile: Named profile controlling defaults (training / application /
            audit / default).
        logger_name: Root logger name. Defaults to the profile default.
        file_name: Log file name override (defaults per profile).
        rotate: Use a :class:`RotatingFileHandler`; when False a plain file
            handler is used.
        marker: Attribute marking handlers owned by this module (idempotency).
        logger: Logger to configure (overrides ``logger_name``).

    Returns:
        The configured logger.
    """
    global _configured

    if profile not in PROFILES:
        raise LoggingConfigurationError(
            f"Unknown logging profile: {profile}",
            detail=profile,
            suggested_resolution=f"Use one of {sorted(PROFILES)}",
        )
    if output not in FORMATS:
        raise LoggingConfigurationError(
            f"Unknown log output format: {output}",
            detail=output,
            suggested_resolution=f"Use one of {sorted(FORMATS)}",
        )

    root = logger or logging.getLogger(logger_name or f"cropfusion.{profile}")
    numeric = logging.getLevelName(level) if isinstance(level, str) else level
    root.setLevel(numeric)

    for handler in list(root.handlers):
        if getattr(handler, marker, False):
            root.removeHandler(handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            _build_formatter(console_output if console_output in FORMATS else "compact")
        )
        console_handler.setLevel(numeric)
        _mark_owned(console_handler, marker)
        root.addHandler(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        name = file_name or PROFILE_FILE_NAMES.get(profile, "cropfusion.log")
        if rotate:
            file_handler = RotatingFileHandler(
                log_path / name,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_path / name, encoding="utf-8")
        file_handler.setFormatter(_build_formatter(output))
        file_handler.setLevel(numeric)
        _mark_owned(file_handler, marker)
        root.addHandler(file_handler)

    root.propagate = False
    _configured = True
    return root


def get_logger(name: str, namespace: str = "cropfusion") -> logging.Logger:
    """Return a namespaced logger under ``namespace``."""
    if name.startswith(namespace + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{namespace}.{name}")


def is_configured() -> bool:
    """True when :func:`setup_logging` has been called."""
    return _configured


def log_dict(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured log entry: ``logger.log(level, event, extra=fields)``."""
    from .formatters import RESERVED

    safe = {k: v for k, v in fields.items() if k not in RESERVED}
    logger.log(level, event, extra=safe)
