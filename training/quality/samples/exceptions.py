"""Exceptions for the sample-quality reporting sub-package."""

from __future__ import annotations

from typing import Any

from shared.exceptions import CropFusionError


class SampleQualityError(CropFusionError):
    """Base class for sample-quality report errors."""

    code: str = "SQ-ERR-001"


class SampleQualityConfigError(SampleQualityError):
    """Raised when sample-quality reporting configuration is invalid."""

    code = "SQ-CONFIG-001"
