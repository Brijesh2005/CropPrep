"""Platform-facing ports: exporter, logger, configuration, serializer, versioning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelExporter(ABC):
    """Port for exporting a trained model into a deployable artifact."""

    @abstractmethod
    def export(self, model: Any, destination: Path, *, format: str = "torch") -> Path:
        """Export ``model`` to ``destination`` and return the artifact path."""

    @abstractmethod
    def export_report(self, destination: Path) -> Path:
        """Write a sidecar report (metrics / metadata) next to the artifact."""


class Logger(ABC):
    """Port for a structured logger factory."""

    @abstractmethod
    def get_logger(self, name: str) -> Any:
        """Return a logger instance for ``name``."""

    @abstractmethod
    def setup(self, **options: Any) -> Any:
        """Apply logging configuration; returns the configured root logger."""


class ConfigurationProvider(ABC):
    """Port for typed configuration access."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Return the setting for ``key`` or ``default``."""

    @abstractmethod
    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load and return the full settings mapping."""


class Serializer(ABC):
    """Port for serializing / deserializing objects."""

    @abstractmethod
    def dump(self, data: Any, path: str | Path) -> Path:
        """Persist ``data`` to ``path`` and return the path."""

    @abstractmethod
    def load(self, path: str | Path) -> Any:
        """Load and return the object stored at ``path``."""


class VersionProvider(ABC):
    """Port for reading and bumping artifact versions."""

    @abstractmethod
    def current(self, name: str) -> Any | None:
        """Return the current version of ``name``, or None when absent."""

    @abstractmethod
    def bump(self, name: str, part: str = "patch", *, message: str = "") -> Any:
        """Bump the current version of ``name`` (major/minor/patch)."""
