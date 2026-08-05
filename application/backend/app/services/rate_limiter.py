"""Rate limiting (in-memory sliding window, Redis optional)."""

from __future__ import annotations

import time
from typing import Any

from app.core.config import RateLimitSettings
from app.core.exceptions import RateLimitError
from app.core.logging import get_logger

logger = get_logger("ratelimit")


class RateLimiter:
    """Checks whether a client has exceeded its request budget."""

    async def allow(self, key: str, limit: int, window_seconds: int) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        pass


class MemoryRateLimiter(RateLimiter):
    """In-memory sliding-window rate limiter (per-process)."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}

    async def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        window = self._hits.setdefault(key, [])
        cutoff = now - window_seconds
        self._hits[key] = [t for t in window if t > cutoff]
        if len(self._hits[key]) >= limit:
            return False
        self._hits[key].append(now)
        return True


class RedisRateLimiter(RateLimiter):
    """Redis fixed-window rate limiter."""

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(redis_url)

    async def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        import time as _t

        bucket = f"ratelimit:{key}:{int(_t.time() // window_seconds)}"
        count = await self._client.incr(bucket)
        if count == 1:
            await self._client.expire(bucket, window_seconds)
        return int(count) <= limit

    async def close(self) -> None:
        await self._client.aclose()


def build_rate_limiter(settings: RateLimitSettings) -> RateLimiter:
    if settings.storage == "redis":
        try:
            return RedisRateLimiter("redis://localhost:6379/0")
        except Exception as exc:  # pragma: no cover
            logger.warning("redis rate limiter unavailable; using memory ({})", exc)
    return MemoryRateLimiter()
