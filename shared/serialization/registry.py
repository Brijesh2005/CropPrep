"""Serialization framework: a registry of named serializers.

A :class:`Serializer` can persist/restore objects in a specific format.
Formats are registered by name and selected through :func:`get_serializer`
or inferred from a file extension via :func:`serializer_for_path`.  Optional
backends (parquet, numpy, torch) are imported lazily so the core framework has
no hard dependency on them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..exceptions import SerializationError, UnsupportedSerializerError


class Serializer(ABC):
    """Contract for a named serialization format."""

    #: Stable format name, e.g. ``"json"``, ``"parquet"``.
    name: str = "unknown"

    #: File extensions this serializer handles (with leading dot).
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def dump(self, data: Any, path: str | Path) -> Path:
        """Persist ``data`` to ``path`` and return the path."""

    @abstractmethod
    def load(self, path: str | Path) -> Any:
        """Load and return the object stored at ``path``."""

    def dumps(self, data: Any) -> bytes:
        """Serialize ``data`` to bytes (default: dump to an in-memory buffer)."""
        import io

        buffer = io.BytesIO()
        temp = Path(buffer.name) if hasattr(buffer, "name") else None
        if temp is None:
            raise SerializationError(
                f"Serializer '{self.name}' does not support in-memory dumps"
            )
        self.dump(data, temp)
        return buffer.getvalue()

    def loads(self, raw: bytes) -> Any:
        """Deserialize ``raw`` bytes (default: load from an in-memory buffer)."""
        import io
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=self.extensions[0] if self.extensions else "") as fh:
            fh.write(raw)
            fh.flush()
            return self.load(fh.name)


class SerializerRegistry:
    """Named registry of :class:`Serializer` implementations."""

    def __init__(self) -> None:
        self._serializers: dict[str, Serializer] = {}
        self._by_extension: dict[str, Serializer] = {}

    def register(self, serializer: Serializer) -> None:
        """Register a serializer under its ``name`` and extensions."""
        self._serializers[serializer.name] = serializer
        for ext in serializer.extensions:
            self._by_extension[ext.lower()] = serializer

    def get(self, name: str) -> Serializer:
        """Return the serializer registered as ``name``."""
        try:
            return self._serializers[name]
        except KeyError:
            raise UnsupportedSerializerError(
                f"No serializer registered for format: {name}",
                detail=name,
                suggested_resolution=f"Register one via SerializerRegistry.register, or use one of {sorted(self._serializers)}",
            ) from None

    def for_path(self, path: str | Path) -> Serializer:
        """Return the serializer registered for ``path``'s extension."""
        ext = Path(path).suffix.lower()
        try:
            return self._by_extension[ext]
        except KeyError:
            raise UnsupportedSerializerError(
                f"No serializer registered for extension: {ext}",
                detail=str(path),
            ) from None

    def names(self) -> list[str]:
        """Names of all registered serializers."""
        return sorted(self._serializers)


#: Process-wide registry with the built-in formats registered.
default_registry = SerializerRegistry()
