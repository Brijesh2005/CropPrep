"""Data access, integrity and storage related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class DataError(CropFusionError):
    """Base class for data-layer failures."""

    code = "CF-DATA-001"


class NotFoundError(DataError):
    """Raised when a requested entity (dataset, file, record) cannot be located."""

    code = "CF-DATA-002"


class IntegrityError(DataError):
    """Raised when a file or record fails integrity checks."""

    code = "CF-DATA-003"


class UnsupportedFormatError(DataError):
    """Raised when a file is not a supported format."""

    code = "CF-DATA-004"


class StorageError(DataError):
    """Raised when a storage backend cannot fulfil a request."""

    code = "CF-DATA-005"


class CacheError(DataError):
    """Raised when a cache backend cannot be read from or written to."""

    code = "CF-DATA-006"


class ProviderError(DataError):
    """Raised when an external data provider fails."""

    code = "CF-DATA-007"


class ScannerError(DataError):
    """Raised when a directory scan cannot complete."""

    code = "CF-DATA-008"
