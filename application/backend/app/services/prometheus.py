"""Prometheus metrics — process, request, inference, cache, model and ML-QA.

Wraps ``prometheus_client`` behind a small typed facade so the rest of the
app never touches the global registry directly. Metrics are namespaced
(``cropfusion_*``) and exposed on ``GET /metrics`` (see ``app.main``).
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)

from app.core.logging import get_logger

logger = get_logger("prometheus")


class PrometheusMetrics:
    """Registry facade for every Phase 11 production metric."""

    def __init__(self, namespace: str = "cropfusion") -> None:
        labels = ["path", "method", "status"]

        self.requests_total = Counter(
            f"{namespace}_requests_total", "HTTP requests", labels
        )
        self.request_duration = Histogram(
            f"{namespace}_request_duration_seconds",
            "HTTP request latency",
            labels,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        self.requests_in_flight = Gauge(
            f"{namespace}_requests_in_flight", "HTTP requests in progress", ["method"]
        )

        self.inference_total = Counter(
            f"{namespace}_inference_total", "Inference executions", ["fallback", "cached"]
        )
        self.inference_duration = Histogram(
            f"{namespace}_inference_duration_seconds",
            "Inference pipeline latency",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )
        self.inference_queue_depth = Gauge(f"{namespace}_inference_queue_depth", "Queued tasks")

        self.cache_requests = Counter(f"{namespace}_cache_requests_total", "Cache lookups")
        self.cache_hits = Counter(f"{namespace}_cache_hits_total", "Cache hits")
        self.cache_hit_ratio = Gauge(f"{namespace}_cache_hit_ratio", "Cache hit ratio")

        self.model_ready = Gauge(f"{namespace}_model_ready", "Model loaded & ready")
        self.model_info = Info(f"{namespace}_model", "Model version")
        self.predictions_total = Counter(
            f"{namespace}_predictions_total", "Predictions served", ["crop"]
        )

        self.active_sessions = Gauge(f"{namespace}_active_sessions", "Active user sessions")
        self.db_query_duration = Histogram(
            f"{namespace}_db_query_duration_seconds",
            "Database query latency",
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        )

        self.drift_severity = Gauge(
            f"{namespace}_drift_severity", "Drift severity per dimension (0/1/2)", ["dimension"]
        )
        self.fairness_status = Gauge(
            f"{namespace}_fairness_status", "Fairness status per attribute (0/1/2)", ["attribute"]
        )
        self.fairness_disparate_impact = Gauge(
            f"{namespace}_fairness_disparate_impact", "Disparate impact ratio", ["attribute"]
        )

    # ------------------------------------------------------------------ #
    # Recording helpers
    # ------------------------------------------------------------------ #

    def record_request(self, path: str, method: str, status: int, duration_ms: float) -> None:
        label = {"path": path, "method": method, "status": str(status)}
        self.requests_total.labels(**label).inc()
        self.request_duration.labels(**label).observe(max(duration_ms / 1000.0, 0.0))

    def start_request(self, method: str) -> None:
        self.requests_in_flight.labels(method=method).inc()

    def finish_request(self, method: str) -> None:
        self.requests_in_flight.labels(method=method).dec()

    def record_inference(self, duration_ms: float, *, fallback: bool, cached: bool) -> None:
        self.inference_total.labels(fallback=str(fallback), cached=str(cached)).inc()
        self.inference_duration.observe(max(duration_ms / 1000.0, 0.0))

    def set_queue_depth(self, depth: int) -> None:
        self.inference_queue_depth.set(max(depth, 0))

    def record_cache(self, hit: bool) -> None:
        self.cache_requests.inc()
        if hit:
            self.cache_hits.inc()

    def update_cache_ratio(self) -> None:
        requests = self.cache_requests._value.get()
        hits = self.cache_hits._value.get()
        self.cache_hit_ratio.set(hits / requests if requests else 0.0)

    def set_model_ready(self, ready: bool, *, version: str | None = None) -> None:
        self.model_ready.set(1 if ready else 0)
        if version:
            self.model_info.info({"version": version})

    def record_prediction(self, crop: str) -> None:
        self.predictions_total.labels(crop=crop).inc()

    def set_active_sessions(self, count: int) -> None:
        self.active_sessions.set(max(count, 0))

    def observe_db_query(self, duration_ms: float) -> None:
        self.db_query_duration.observe(max(duration_ms / 1000.0, 0.0))

    def set_drift(self, dimension: str, severity: str) -> None:
        self.drift_severity.labels(dimension=dimension).set(_severity_code(severity))

    def set_fairness(self, attribute: str, status: str) -> None:
        self.fairness_status.labels(attribute=attribute).set(_fairness_code(status))

    def set_disparate_impact(self, attribute: str, ratio: float) -> None:
        self.fairness_disparate_impact.labels(attribute=attribute).set(max(ratio, 0.0))

    # ------------------------------------------------------------------ #
    # Export / scrape
    # ------------------------------------------------------------------ #

    def render(self) -> bytes:
        """Return the Prometheus text exposition (``text/plain`` body)."""
        return generate_latest()

    @staticmethod
    def content_type() -> str:
        return CONTENT_TYPE_LATEST

    def expose_http(self, port: int) -> None:
        """Serve metrics on a sidecar HTTP port (debug / local use)."""
        start_http_server(port)


def _severity_code(severity: str) -> int:
    return {"low": 0, "moderate": 1, "high": 2}.get(severity.lower(), 0)


def _fairness_code(status: str) -> int:
    return {"compliant": 0, "at_risk": 1, "violating": 2}.get(status.lower(), 1)


#: Default singleton used by the middleware and the ``/metrics`` endpoint.
metrics = PrometheusMetrics()
