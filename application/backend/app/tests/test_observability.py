"""Phase 11 observability tests — Prometheus registry, middleware, metrics registry, tracing."""

from __future__ import annotations

import uuid

import pytest

from app.services.metrics import MetricsRegistry
from app.services.prometheus import PrometheusMetrics


@pytest.fixture
def isolated_metrics():
    """A metrics registry on a unique namespace so tests never collide on the default collector."""
    return PrometheusMetrics(namespace=f"test_{uuid.uuid4().hex[:8]}")


# --------------------------------------------------------------------------- #
# PrometheusMetrics registry facade
# --------------------------------------------------------------------------- #


def test_record_request_increments_counter_and_histogram(isolated_metrics):
    isolated_metrics.record_request("/api/v1/predict", "POST", 200, 12.3)
    labels = isolated_metrics.requests_total.labels(path="/api/v1/predict", method="POST", status="200")
    assert labels._value.get() == 1.0
    duration = isolated_metrics.request_duration.labels(path="/api/v1/predict", method="POST", status="200")
    assert duration._sum.get() == pytest.approx(0.0123)
    samples = duration.collect()[0].samples
    inf_bucket = next(s for s in samples if s.name.endswith("_bucket") and s.labels.get("le") == "+Inf")
    assert inf_bucket.value == 1.0


def test_in_flight_gauge_round_trips(isolated_metrics):
    isolated_metrics.start_request("GET")
    assert isolated_metrics.requests_in_flight.labels(method="GET")._value.get() == 1.0
    isolated_metrics.finish_request("GET")
    assert isolated_metrics.requests_in_flight.labels(method="GET")._value.get() == 0.0


def test_record_inference_marks_fallback_and_cached(isolated_metrics):
    isolated_metrics.record_inference(5.0, fallback=True, cached=False)
    label = isolated_metrics.inference_total.labels(fallback="True", cached="False")
    assert label._value.get() == 1.0


def test_cache_ratio_updated_from_hits(isolated_metrics):
    isolated_metrics.record_cache(True)
    isolated_metrics.record_cache(False)
    isolated_metrics.record_cache(True)
    isolated_metrics.update_cache_ratio()
    assert isolated_metrics.cache_hit_ratio._value.get() == pytest.approx(2 / 3)
    assert isolated_metrics.cache_requests._value.get() == 3.0


def test_model_ready_gauge_and_info(isolated_metrics):
    isolated_metrics.set_model_ready(True, version="1.2.3")
    assert isolated_metrics.model_ready._value.get() == 1.0
    assert isolated_metrics.model_info._value == {"version": "1.2.3"}
    isolated_metrics.set_model_ready(False)
    assert isolated_metrics.model_ready._value.get() == 0.0


def test_ml_quality_gauges_map_severity_codes(isolated_metrics):
    isolated_metrics.set_drift("features", "high")
    assert isolated_metrics.drift_severity.labels(dimension="features")._value.get() == 2.0
    isolated_metrics.set_fairness("region", "violating")
    assert isolated_metrics.fairness_status.labels(attribute="region")._value.get() == 2.0
    isolated_metrics.set_disparate_impact("region", 0.62)
    assert isolated_metrics.fairness_disparate_impact.labels(attribute="region")._value.get() == pytest.approx(0.62)


def test_render_returns_prometheus_text_exposition(isolated_metrics):
    isolated_metrics.set_model_ready(True)
    body = isolated_metrics.render()
    assert isolated_metrics.content_type().startswith("text/plain")
    text = body.decode("utf-8")
    assert f"# HELP {isolated_metrics.model_ready._name}" in text
    assert "# TYPE" in text


# --------------------------------------------------------------------------- #
# PrometheusMiddleware
# --------------------------------------------------------------------------- #


def _minimal_app(metrics):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse

    from app.middleware.prometheus import PrometheusMiddleware

    app = Starlette()
    app.add_middleware(PrometheusMiddleware, metrics=metrics)

    @app.route("/hello")
    async def hello(request):
        return JSONResponse({"ok": True})

    @app.route("/boom")
    async def boom(request):
        raise RuntimeError("boom")

    @app.route("/health")
    async def health(request):
        return JSONResponse({"status": "ok"})

    return app


def test_middleware_records_request_and_returns_in_flight_to_zero(isolated_metrics):
    from starlette.testclient import TestClient

    with TestClient(_minimal_app(isolated_metrics)) as client:
        assert client.get("/hello").status_code == 200

    labels = isolated_metrics.requests_total.labels(path="/hello", method="GET", status="200")
    assert labels._value.get() == 1.0
    assert isolated_metrics.requests_in_flight.labels(method="GET")._value.get() == 0.0


def test_middleware_records_five_hundred_on_exception(isolated_metrics):
    from starlette.testclient import TestClient

    with TestClient(_minimal_app(isolated_metrics), raise_server_exceptions=False) as client:
        assert client.get("/boom").status_code == 500

    labels = isolated_metrics.requests_total.labels(path="/boom", method="GET", status="500")
    assert labels._value.get() == 1.0
    # The in-flight gauge must be balanced even when the handler raises.
    assert isolated_metrics.requests_in_flight.labels(method="GET")._value.get() == 0.0


def test_middleware_skips_health_paths(isolated_metrics):
    from starlette.testclient import TestClient

    with TestClient(_minimal_app(isolated_metrics)) as client:
        assert client.get("/health").status_code == 200

    labels = isolated_metrics.requests_total.labels(path="/health", method="GET", status="200")
    assert labels._value.get() == 0.0
    assert isolated_metrics.requests_in_flight.labels(method="GET")._value.get() == 0.0


# --------------------------------------------------------------------------- #
# MetricsRegistry (runtime snapshot for the monitoring module)
# --------------------------------------------------------------------------- #


def test_metrics_registry_snapshot_aggregates():
    registry = MetricsRegistry()
    registry.record("/api/v1/predict", 200, 10.0)
    registry.record("/api/v1/predict", 500, 20.0)

    snapshot = registry.snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["errors"] == 1
    assert snapshot["avg_latency_ms"] == pytest.approx(15.0)
    assert snapshot["requests_per_second"] >= 0
    assert snapshot["by_path"]["/api/v1/predict"]["requests"] == 2
    assert snapshot["by_path"]["/api/v1/predict"]["avg_ms"] == pytest.approx(15.0)
    assert snapshot["by_path"]["/api/v1/predict"]["errors"] == 1


def test_metrics_registry_empty_snapshot():
    snapshot = MetricsRegistry().snapshot()
    assert snapshot["requests"] == 0
    assert snapshot["errors"] == 0
    assert snapshot["avg_latency_ms"] == 0.0
    assert snapshot["by_path"] == {}


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #


def test_prometheus_endpoint_serves_exposition(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "cropfusion_requests_total" in r.text


def test_prometheus_endpoint_absent_when_disabled(settings):
    from starlette.testclient import TestClient

    from app.main import create_app

    settings.monitoring.prometheus_enabled = False
    with TestClient(create_app(settings)) as client:
        assert client.get("/metrics").status_code == 404


def test_prometheus_records_requests_end_to_end(client):
    before = __import__("app.services.prometheus", fromlist=["metrics"]).metrics
    client.get("/api/v1/config")
    rendered = before.render().decode("utf-8")
    assert 'cropfusion_requests_total{method="GET",path="/api/v1/config"' in rendered


# --------------------------------------------------------------------------- #
# OpenTelemetry tracing
# --------------------------------------------------------------------------- #


def test_setup_tracing_noop_when_disabled(settings):
    from fastapi import FastAPI

    from app.core.tracing import setup_tracing

    settings.monitoring.tracing_enabled = False
    app = FastAPI()
    setup_tracing(app, settings)  # must not raise or instrument


def test_setup_tracing_console_exporter_runs(settings):
    from fastapi import FastAPI

    from app.core.tracing import setup_tracing

    settings.monitoring.tracing_enabled = True
    settings.monitoring.tracing_exporter = "console"
    app = FastAPI()
    setup_tracing(app, settings)  # instruments without raising when SDK is present


def test_setup_tracing_otlp_degrades_gracefully(settings):
    from fastapi import FastAPI

    from app.core.tracing import setup_tracing

    settings.monitoring.tracing_enabled = True
    settings.monitoring.tracing_exporter = "otlp"
    app = FastAPI()
    setup_tracing(app, settings)  # optional dependency missing -> warning, not an exception
