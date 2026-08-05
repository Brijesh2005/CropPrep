"""Logging related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class LoggingConfigurationError(CropFusionError):
    """Raised when a logging setup cannot be applied."""

    code = "CF-LOG-001"
