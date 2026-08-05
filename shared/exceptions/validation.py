"""Validation related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class ValidationFailedError(CropFusionError):
    """Raised when validation fails (blocking errors present)."""

    code = "CF-VALID-001"


class ValidationNotSupportedError(CropFusionError):
    """Raised when no validator is registered for a target type."""

    code = "CF-VALID-002"
