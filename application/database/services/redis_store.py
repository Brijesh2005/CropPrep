"""Redis integration for the enterprise data layer.

:class:`RedisStore` is a thin namespaced key/value facade over ``redis.asyncio``
used for session caching, prediction caching, spatial caching, temporary reports
and rate limiting. When Redis is disabled or unavailable it falls back to an
in-memory store so the application remains fully functional in development and
in the test-suite (which injects ``fakeredis``).
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import RedisSettings
from app.core.logging import get_logger

logger = get_logger("redis-store")


class RedisStore:
    """Namespaced async key/value store (Redis-backed)."""

    def __init__(self, settings: RedisSettings, client: Any | None = None) -> None:
        self._settings = settings
        if client is not None:
            self._client = client
        else:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(settings.url)

    def _key(self, key: str) -> str:
        return f"{self._settings.key_prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        raw = await self._client.get(self._key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        import json

        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        import json

        payload = json.dumps(value, default=str)
        ttl = self._settings.session_ttl_seconds if ttl is None else ttl
        await self._client.set(self._key(key), payload, ex=ttl if ttl and ttl > 0 else None)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(self._key(key)))

    async def ttl(self, key: str) -> int:
        return int(await self._client.ttl(self._key(key)))

    async def expire(self, key: str, ttl: int) -> None:
        await self._client.expire(self._key(key), ttl)

    async def incr(self, key: str, amount: int = 1) -> int:
        return int(await self._client.incr(self._key(key), amount))

    async def keys(self, pattern: str = "*") -> list[str]:
        found = await self._client.keys(self._key(pattern))
        return [str(k) for k in found]

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except AttributeError:  # pragma: no cover - fakeredis variants
            pass


class MemoryStore(RedisStore):
    """In-memory fallback implementing the same interface (dev/tests)."""

    def __init__(self, settings: RedisSettings, client: Any | None = None) -> None:
        self._settings = settings
        self._data: dict[str, tuple[Any, float | None]] = {}

    def _key(self, key: str) -> str:
        return f"{self._settings.key_prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        entry = self._data.get(self._key(key))
        if entry is None:
            return None
        value, expires = entry
        if expires is not None and time.monotonic() > expires:
            self._data.pop(self._key(key), None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = self._settings.session_ttl_seconds if ttl is None else ttl
        expires = time.monotonic() + ttl if ttl and ttl > 0 else None
        self._data[self._key(key)] = (value, expires)

    async def delete(self, key: str) -> None:
        self._data.pop(self._key(key), None)

    async def exists(self, key: str) -> bool:
        return self._key(key) in self._data

    async def ttl(self, key: str) -> int:
        entry = self._data.get(self._key(key))
        if entry is None or entry[1] is None:
            return -1
        return max(int(entry[1] - time.monotonic()), 0)

    async def expire(self, key: str, ttl: int) -> None:
        entry = self._data.get(self._key(key))
        if entry is not None:
            self._data[self._key(key)] = (entry[0], time.monotonic() + ttl)

    async def incr(self, key: str, amount: int = 1) -> int:
        k = self._key(key)
        entry = self._data.get(k)
        if entry is None:
            self._data[k] = (amount, None)
            return amount
        value, expires = entry
        value = (value if isinstance(value, int) else 0) + amount
        self._data[k] = (value, expires)
        return value

    async def keys(self, pattern: str = "*") -> list[str]:
        prefix = self._key("")
        return [k[len(prefix):] for k in self._data if k.startswith(prefix)]

    async def close(self) -> None:
        self._data.clear()


def build_redis_store(settings: RedisSettings, client: Any | None = None) -> RedisStore:
    """Build the configured store (graceful fallback to memory)."""
    if settings.enabled or client is not None:
        try:
            return RedisStore(settings, client=client)
        except Exception as exc:  # pragma: no cover - redis optional
            logger.warning("redis unavailable; using memory store ({})", exc)
    return MemoryStore(settings)
