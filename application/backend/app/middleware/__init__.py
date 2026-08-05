"""HTTP middleware: request-id, logging, timing, rate limit, security headers."""

from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from app.core.config import Settings
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.prometheus import PrometheusMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.middleware.timing import PerformanceMiddleware


def register_middleware(app: FastAPI, settings: Settings, container: object) -> None:
    """Register every middleware on ``app`` in the right order (outer → inner)."""

    app.add_middleware(CORSMiddleware, **settings.cors.model_dump())
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if settings.monitoring.prometheus_enabled:
        app.add_middleware(PrometheusMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, settings=settings, container=container)
    app.add_middleware(RequestIDMiddleware, header=settings.log.correlation_header)
    app.add_middleware(PerformanceMiddleware, container=container)
    app.add_middleware(RequestLoggingMiddleware, header=settings.log.correlation_header)
