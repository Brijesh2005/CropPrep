"""Rate-limiting middleware (in-memory sliding window)."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import Settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce ``rate_limit.requests_per_minute`` per client IP."""

    def __init__(self, app, settings: Settings, container) -> None:
        super().__init__(app)
        self._settings = settings
        self._container = container

    async def dispatch(self, request: Request, call_next):
        if not self._settings.rate_limit.enabled:
            return await call_next(request)
        limiter = self._limiter()
        if limiter is None:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        key = f"{request.url.path}:{client}"
        allowed = await limiter.allow(
            key, self._settings.rate_limit.requests_per_minute, 60
        )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "B-RATE-001",
                        "message": "rate limit exceeded, try again shortly",
                        "status": 429,
                    }
                },
            )
        return await call_next(request)

    def _limiter(self):
        try:
            return self._container.services.resolve("rate_limiter")
        except Exception:
            return None
