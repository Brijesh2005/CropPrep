"""Serialization and file I/O related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class SerializationError(CropFusionError):
    """Raised when an object cannot be serialized or deserialized."""

    code = "CF-IO-001"


class UnsupportedSerializerError(SerializationError):
    """Raised when no serializer is registered for a requested format."""

    code = "CF-IO-002"


class FileAccessError(SerializationError):
    """Raised when a file cannot be opened, read or written."""

    code = "CF-IO-003"
