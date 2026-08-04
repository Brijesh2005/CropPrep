"""Performance timing middleware — records latency into the metrics registry."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Record request latency (ms) into the shared metrics registry."""

    def __init__(self, app, container) -> None:
        super().__init__(app)
        self._container = container

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics = self._metrics()
            if metrics is not None:
                metrics.record(request.url.path, 500, (time.perf_counter() - start) * 1000)
            raise
        if self._metrics() is not None:
            self._metrics().record(
                request.url.path, response.status_code, (time.perf_counter() - start) * 1000
            )
        return response

    def _metrics(self):
        try:
            return self._container.services.resolve("metrics")
        except Exception:
            return None
