"""Structured logging for the Spatial-Temporal Alignment Module (STAM).

Reuses the formatters and setup from the Dataset Manager logger so that STAM
and the DMS share one consistent, JSON-capable logging configuration without
duplicating formatter code. STAM loggers live under
``cropfusion.spatial_alignment.*``.
"""

from __future__ import annotations

import logging

from shared.logging.formatters import CompactFormatter, JsonFormatter

ROOT_NAME = "cropfusion.spatial_alignment"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger("stam")``."""
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
    """Configure the STAM logger hierarchy.

    Args:
        level: Log level name or numeric level.
        log_dir: Optional directory for rotating JSON log files.
        max_bytes: Rotation size cap.
        backup_count: Rotated files retained.
        json_format: JSON file formatter (console stays compact).
        console: Install a stderr console handler.
        logger: Logger to configure (defaults to the STAM root).

    Returns:
        The configured logger.
    """
    from logging.handlers import RotatingFileHandler
    import sys

    root = logger or get_logger("")
    numeric = logging.getLevelName(level) if isinstance(level, str) else level
    root.setLevel(numeric)

    for handler in list(root.handlers):
        if getattr(handler, "_stam_owned", False):
            root.removeHandler(handler)

    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(CompactFormatter())
        handler.setLevel(numeric)
        setattr(handler, "_stam_owned", True)
        root.addHandler(handler)

    if log_dir:
        from pathlib import Path

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            Path(log_dir) / "stam.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter() if json_format else CompactFormatter())
        handler.setLevel(numeric)
        setattr(handler, "_stam_owned", True)
        root.addHandler(handler)

    root.propagate = False
    return root
