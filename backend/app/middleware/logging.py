"""Structured request logging middleware."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logging import get_logger, get_correlation_id

logger = get_logger("http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log a structured line per request (method, path, status, latency)."""

    def __init__(self, app, header: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header = header

    async def dispatch(self, request: Request, call_next):
        import time

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.bind(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=get_correlation_id(),
        ).info("request")
        return response
