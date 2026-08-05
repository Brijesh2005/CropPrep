"""Custom exception hierarchy for the Dataset Manager package.

Every error raised by the Dataset Manager is an instance of
:class:`DatasetManagerError` (or one of its subclasses). This lets callers
catch a single base type while still discriminating between failure modes
using the more specific subclasses and their stable machine-readable error
codes.

Error codes follow the convention ``DM-<AREA>-<NNN>`` so that automated
consumers (CLI, REST adapters, dashboards) can map errors deterministically
without string matching.
"""

from __future__ import annotations

from typing import Any

from shared.exceptions import CropFusionError


class DatasetManagerError(CropFusionError):
    """Base class for all Dataset Manager errors.

    Attributes:
        code: Stable machine-readable error code, e.g. ``"DM-DL-001"``.
        message: Human readable description of the failure.
        detail: Optional structured detail (offending path, expected value,
            actual value, ...) attached to the error.
        suggested_resolution: Optional human readable guidance for recovering.
    """

    code: str = "DM-ERROR"


class InvalidConfigurationError(DatasetManagerError):
    """Raised when configuration is malformed, unknown or out of range."""

    code = "DM-CONFIG-001"


class DatasetNotFoundError(DatasetManagerError):
    """Raised when a requested dataset (or dataset root) cannot be located."""

    code = "DM-FIND-001"


class CorruptedDatasetError(DatasetManagerError):
    """Raised when a file fails integrity checks (header parse, checksum, ...)."""

    code = "DM-CORRUPT-001"


class InvalidMetadataError(DatasetManagerError):
    """Raised when metadata is missing, malformed or inconsistent."""

    code = "DM-META-001"


class ValidationFailedError(DatasetManagerError):
    """Raised when a dataset fails validation (blocking errors present)."""

    code = "DM-VALID-001"


class DownloadFailedError(DatasetManagerError):
    """Raised when a dataset download (or its materialization) fails."""

    code = "DM-DL-001"


class CacheError(DatasetManagerError):
    """Raised when the cache backend cannot be read from or written to."""

    code = "DM-CACHE-001"


class RegistryError(DatasetManagerError):
    """Raised when the dataset registry cannot fulfil a request."""

    code = "DM-REG-001"


class UnsupportedFormatError(DatasetManagerError):
    """Raised when a file is not a supported format (CSV / GeoTIFF)."""

    code = "DM-FMT-001"


class ScannerError(DatasetManagerError):
    """Raised when the directory scanner cannot complete a scan."""

    code = "DM-SCAN-001"
