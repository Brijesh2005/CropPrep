"""Structured logging shared across the CropFusion platforms.

Consolidates the formatters and setup logic that were previously duplicated
across the training packages (``cropfusion.dataset_manager``,
``cropfusion.spatial_alignment``, ...) and exposes profiles for Training,
Application and audit logging.
"""

from __future__ import annotations

from .audit import audit
from .formatters import (
    ColoredFormatter,
    CompactFormatter,
    JsonFormatter,
    RESERVED,
)
from .setup import (
    get_logger,
    is_configured,
    log_dict,
    setup_logging,
)

__all__ = [
    "ColoredFormatter",
    "CompactFormatter",
    "JsonFormatter",
    "RESERVED",
    "audit",
    "get_logger",
    "is_configured",
    "log_dict",
    "setup_logging",
]
