"""Structured logging for the dataset export package.

Loggers live under ``cropfusion.export.*``.
"""

from __future__ import annotations

import logging

ROOT_NAME = "cropfusion.export"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger("builder")``."""
    if name.startswith(ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
