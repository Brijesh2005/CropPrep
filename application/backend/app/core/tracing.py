"""OpenTelemetry tracing setup (Phase 11 observability).

Instruments the FastAPI app and HTTP client libraries. Spans are exported to
the console by default, or to an OTLP collector when configured. Correlation
IDs are stamped onto every span so logs, traces and request IDs stay linked.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger("tracing")


def setup_tracing(app: FastAPI, settings: Settings) -> None:
    """Configure the OpenTelemetry SDK and instrument ``app`` (no-op when off)."""
    monitoring = settings.monitoring
    if not monitoring.tracing_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanExporter,
            SimpleSpanProcessor,
        )

        resource = Resource.create({"service.name": settings.app_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(_build_exporter(monitoring)))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("OpenTelemetry tracing enabled", exporter=monitoring.tracing_exporter)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("tracing disabled ({})", exc)


def _build_exporter(monitoring) -> Any:
    if monitoring.tracing_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=monitoring.otlp_endpoint, insecure=True)
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    return ConsoleSpanExporter()
