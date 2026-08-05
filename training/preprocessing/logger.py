"""Structured logging for the preprocessing pipeline.

Reuses the Dataset Manager / STAM formatters so the whole platform shares one
JSON-capable logging configuration.
"""

from __future__ import annotations

import logging

from shared.logging.formatters import CompactFormatter, JsonFormatter

ROOT_NAME = "cropfusion.preprocessing"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger("tabular")``."""
    if name.startswith(ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
