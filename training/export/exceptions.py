"""Exceptions for the dataset export package."""

from __future__ import annotations

from shared.exceptions import CropFusionError


class ExportError(CropFusionError):
    """Base class for dataset-export errors."""

    code: str = "EX-ERROR"


class ExportConfigError(ExportError):
    """Raised when export configuration is invalid."""

    code = "EX-CONFIG-001"


class ExportFormatError(ExportError):
    """Raised when an unknown or unsupported format is requested."""

    code = "EX-FORMAT-001"


class ExportWriteError(ExportError):
    """Raised when an artifact cannot be written."""

    code = "EX-WRITE-001"
