"""Cache abstraction with an in-memory fallback and optional Redis backend."""

from __future__ import annotations

import time
from typing import Any

from app.core.config import CacheSettings
from app.core.logging import get_logger

logger = get_logger("cache")


class Cache:
    """Async key/value cache interface."""

    async def get(self, key: str) -> Any | None:  # pragma: no cover - interface
        raise NotImplementedError

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        pass


class MemoryCache(Cache):
    """Thread-safe in-memory cache (dev / single-process)."""

    def __init__(self, default_ttl: int = 3600) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires = entry
        if expires is not None and time.monotonic() > expires:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = self._default_ttl if ttl is None else ttl
        expires = time.monotonic() + ttl if ttl and ttl > 0 else None
        self._store[key] = (value, expires)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


class RedisCache(Cache):
    """Redis-backed cache (activated when a Redis client is available)."""

    def __init__(self, redis_url: str, default_ttl: int = 3600) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(redis_url)
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        import json

        try:
            return json.loads(raw)
        except Exception:
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        import json

        ttl = self._default_ttl if ttl is None else ttl
        await self._client.set(key, json.dumps(value, default=str), ex=ttl)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def close(self) -> None:
        await self._client.aclose()


def build_cache(settings: CacheSettings) -> Cache:
    """Build the configured cache backend (graceful Redis fallback)."""
    if settings.backend == "redis":
        try:
            return RedisCache(settings.redis_url, settings.default_ttl_seconds)
        except Exception as exc:  # pragma: no cover - redis optional
            logger.warning("redis unavailable; using memory cache ({})", exc)
    return MemoryCache(settings.default_ttl_seconds)
