"""Repository / Cache / Storage ports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Repository(ABC):
    """Port for a persisted collection of records keyed by an identifier."""

    @abstractmethod
    def save(self, record: Any) -> Any:
        """Insert or replace ``record``; return its identifier."""

    @abstractmethod
    def save_many(self, records: list[Any]) -> int:
        """Bulk upsert; return the number of records written."""

    @abstractmethod
    def get(self, key: Any) -> Any | None:
        """Fetch a record by ``key``, or None when absent."""

    @abstractmethod
    def query(self, **filters: Any) -> list[Any]:
        """Return records matching ``filters``."""

    @abstractmethod
    def count(self) -> int:
        """Total number of records."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class Cache(ABC):
    """Port for a key/value cache with TTL and prefix invalidation."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the deserialised value for ``key``, or None on miss/expiry."""

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove ``key``; returns True when it existed."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Remove all keys starting with ``prefix``; returns the count."""

    @abstractmethod
    def clear(self) -> int:
        """Drop all entries; returns the count removed."""

    @abstractmethod
    def prune(self) -> int:
        """Remove expired entries; returns the count removed."""


class Storage(ABC):
    """Port for a filesystem-like storage backend."""

    @abstractmethod
    def exists(self, path: Path) -> bool:
        """True when ``path`` exists."""

    @abstractmethod
    def read_bytes(self, path: Path) -> bytes:
        """Read the raw bytes at ``path``."""

    @abstractmethod
    def write_bytes(self, path: Path, data: bytes) -> Path:
        """Write raw bytes to ``path`` and return it."""

    @abstractmethod
    def delete(self, path: Path) -> bool:
        """Remove ``path``; returns True when it existed."""

    @abstractmethod
    def list(self, root: Path) -> list[Path]:
        """Return all paths under ``root``."""
