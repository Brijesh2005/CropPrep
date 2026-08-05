"""Serialization framework shared across the CropFusion platforms.

Provides a pluggable registry of serializers for JSON, YAML, pickle, parquet,
CSV, NumPy and PyTorch formats.  Optional-backend formats degrade gracefully
with a helpful error when the dependency is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import SerializationError, UnsupportedSerializerError
from .formats import register_builtins
from .registry import Serializer, SerializerRegistry, default_registry

register_builtins(default_registry)


def get_serializer(name: str) -> Serializer:
    """Return the serializer registered under ``name``."""
    return default_registry.get(name)


def serializer_for_path(path: str | Path) -> Serializer:
    """Return the serializer matching ``path``'s file extension."""
    return default_registry.for_path(path)


def dump(data: Any, path: str | Path) -> Path:
    """Serialize ``data`` to ``path``, choosing the format by extension."""
    return serializer_for_path(path).dump(data, path)


def load(path: str | Path) -> Any:
    """Load an object from ``path``, choosing the format by extension."""
    return serializer_for_path(path).load(path)


__all__ = [
    "CsvSerializer",
    "JsonSerializer",
    "NumpySerializer",
    "ParquetSerializer",
    "PickleSerializer",
    "SerializationError",
    "Serializer",
    "SerializerRegistry",
    "TorchSerializer",
    "UnsupportedSerializerError",
    "YamlSerializer",
    "default_registry",
    "dump",
    "get_serializer",
    "load",
    "register_builtins",
    "serializer_for_path",
]

from .formats import (  # noqa: E402
    CsvSerializer,
    JsonSerializer,
    NumpySerializer,
    ParquetSerializer,
    PickleSerializer,
    TorchSerializer,
    YamlSerializer,
)
