"""Runtime cache (Phase R6).

A small thread-safe LRU cache with byte accounting used by the loaders to keep
repeated metadata lookups (parquet dataframes, query results) cheap while
honouring the runtime memory limits.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Hashable


def _estimate_size(value: Any) -> int:
    """Best-effort byte estimate for a cached value."""
    if hasattr(value, "memory_usage") and callable(value.memory_usage):
        try:
            return int(value.memory_usage(deep=True).sum())
        except Exception:  # pragma: no cover - non-tabular container
            pass
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except Exception:  # pragma: no cover
            pass
    try:
        import sys

        return sys.getsizeof(value)
    except TypeError:  # pragma: no cover
        return 0


class RuntimeCache:
    """Thread-safe LRU cache with a byte budget and optional TTL.

    Args:
        max_bytes: Hard byte budget (0 disables byte eviction).
        max_entries: Maximum number of entries (0 disables entry eviction).
        ttl_seconds: Optional per-entry time-to-live (``None`` = no TTL).
    """

    def __init__(
        self,
        max_bytes: int = 256 * 1024 * 1024,
        max_entries: int = 256,
        ttl_seconds: int | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._data: dict[Hashable, Any] = {}
        self._sizes: dict[Hashable, int] = {}
        self._added: dict[Hashable, float] = {}
        self._total_bytes = 0
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get(self, key: Hashable) -> Any:
        """Return the cached value or ``None``."""
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return None
            if self.ttl_seconds is not None:
                if time.monotonic() - self._added[key] > self.ttl_seconds:
                    self._drop(key)
                    self._misses += 1
                    return None
            self._hits += 1
            return self._data[key]

    def set(self, key: Hashable, value: Any) -> None:
        """Insert / refresh a value, evicting until within budget."""
        with self._lock:
            if key in self._data:
                self._drop(key)
            size = _estimate_size(value)
            self._data[key] = value
            self._sizes[key] = size
            self._added[key] = time.monotonic()
            self._total_bytes += size
            self._evict_until_under_budget()

    def contains(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._data

    def evict(self) -> int:
        """Evict least-recently-used entries, returning freed bytes."""
        with self._lock:
            if not self._data:
                return 0
            # A single-pass LRU approximation: drop the oldest additions.
            oldest = sorted(self._added, key=self._added.get)
            freed = 0
            for key in oldest:
                freed += self._drop(key)
            return freed

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._sizes.clear()
            self._added.clear()
            self._total_bytes = 0

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._data),
                "bytes": self._total_bytes,
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _drop(self, key: Hashable) -> int:
        size = self._sizes.pop(key, 0)
        self._data.pop(key, None)
        self._added.pop(key, None)
        self._total_bytes -= size
        return size

    def _evict_until_under_budget(self) -> None:
        while self._data and self._total_bytes > self.max_bytes:
            oldest = min(self._added, key=self._added.get)
            self._drop(oldest)
        while self._data and len(self._data) > self.max_entries:
            oldest = min(self._added, key=self._added.get)
            self._drop(oldest)
