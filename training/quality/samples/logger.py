"""Structured logging for the sample-quality reporting sub-package.

Loggers live under ``cropfusion.quality.samples.*`` and reuse the shared
formatters.
"""

from __future__ import annotations

import logging

ROOT_NAME = "cropfusion.quality.samples"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger("samples")``."""
    if name.startswith(ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
