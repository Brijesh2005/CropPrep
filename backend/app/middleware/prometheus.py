"""Prometheus HTTP middleware — records request + response telemetry."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.services.prometheus import PrometheusMetrics, metrics as default_metrics

_SKIP_PATHS = {"/metrics", "/health", "/ready", "/live"}


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count, latency histogram and in-flight gauge."""

    def __init__(self, app, metrics: PrometheusMetrics | None = None) -> None:
        super().__init__(app)
        self._metrics = metrics or default_metrics

    @property
    def metrics(self) -> PrometheusMetrics:
        return self._metrics

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        if path in _SKIP_PATHS:
            return await call_next(request)

        self._metrics.start_request(method)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            self._metrics.record_request(path, method, 500, (time.perf_counter() - start) * 1000)
            raise
        finally:
            self._metrics.finish_request(method)
        self._metrics.record_request(path, method, response.status_code, (time.perf_counter() - start) * 1000)
        return response
