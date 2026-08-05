"""STAM caching on top of the Dataset Manager cache.

Everything is keyed under the ``stam:`` namespace so STAM cache entries can be
cleared independently of Dataset Manager entries. Cached artefacts:

* spatial index / boundary index (built lazily once),
* nearest-location results,
* assembled observations (per location/year/season),
* resolved temporal contexts.

Because the underlying store is the Dataset Manager's SQLite cache, entries
survive process restarts and respect TTLs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .interfaces import StamCache
from .logger import get_logger

logger = get_logger("cache")

_NAMESPACE = "stam:"


class DatasetManagerStamCache(StamCache):
    """Cache adapter backed by the Dataset Manager's cache interface."""

    def __init__(
        self,
        manager: Any,
        *,
        enabled: bool = True,
        default_ttl_seconds: int = 3600,
    ) -> None:
        self.manager = manager
        self.enabled = enabled
        self.default_ttl = default_ttl_seconds

    # -- Key helpers ---------------------------------------------------------- #

    @staticmethod
    def observation_key(lon: float, lat: float, year: int, season: str | None) -> str:
        digest = hashlib.sha256(f"{lon:.5f}|{lat:.5f}|{year}|{season}".encode()).hexdigest()[:16]
        return f"{_NAMESPACE}obs:{digest}"

    @staticmethod
    def nearest_key(lon: float, lat: float) -> str:
        return f"{_NAMESPACE}nearest:{lon:.5f}:{lat:.5f}"

    @staticmethod
    def temporal_key(year: int | None, season: str | None, reference: str | None) -> str:
        return f"{_NAMESPACE}temporal:{year}:{season}:{reference or 'none'}"

    @staticmethod
    def index_key(name: str) -> str:
        return f"{_NAMESPACE}index:{name}"

    # -- Interface ------------------------------------------------------------ #

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        value = self.manager.cache_get(key)
        if value is None:
            return None
        return value

    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        if not self.enabled:
            return
        self.manager.cache_set(key, value, ttl_seconds=ttl_seconds or self.default_ttl)

    def delete(self, key: str) -> bool:
        if not self.enabled:
            return False
        return bool(self.manager.cache_invalidate(key))

    def clear(self, prefix: str | None = None) -> int:
        if not self.enabled:
            return 0
        if prefix is None:
            return int(self.manager.cache_invalidate(_NAMESPACE))
        return int(self.manager.cache_invalidate(f"{_NAMESPACE}{prefix}"))
