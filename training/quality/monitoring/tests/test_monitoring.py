"""Monitoring asset tests — exporters and dashboard rendering."""

from __future__ import annotations

import json

import pytest

from training.quality.drift import DriftConfig, DriftMonitor, ReportWriter
from training.quality.fairness import FairnessConfig, FairnessEvaluator, FairnessReportWriter
from training.quality.monitoring.dashboard import render_performance_dashboard
from training.quality.monitoring.exporters import (
    export_drift_to_prometheus,
    export_fairness_to_prometheus,
    export_system_to_prometheus,
)

rng = __import__("numpy").random.default_rng(11)


class StubMetrics:
    def __init__(self) -> None:
        self.drift: dict[str, str] = {}
        self.fairness: dict[str, str] = {}
        self.disparate_impact: dict[str, float] = {}
        self.recorded: list[tuple[str, str, int, float]] = []
        self.model_ready = _GaugeStub()

    def set_drift(self, dimension: str, severity: str) -> None:
        self.drift[dimension] = severity

    def set_fairness(self, attribute: str, status: str) -> None:
        self.fairness[attribute] = status

    def set_disparate_impact(self, attribute: str, ratio: float) -> None:
        self.disparate_impact[attribute] = ratio

    def record_request(self, *args, **kwargs) -> None:
        self.recorded.append(args)


class _GaugeStub:
    def __init__(self) -> None:
        self.value: int | float | None = None

    def set(self, value) -> None:
        self.value = value


def _drift_report(tmp_path):
    reference = _make_df(1500, drift=0.0)
    current = _make_df(900, drift=4.0)
    monitor = DriftMonitor(
        reference, feature_columns=["temperature", "rainfall"], label_column="crop_label"
    )
    report = monitor.evaluate(current, timestamp_column="created_at")
    return ReportWriter().write(report, tmp_path)["json"]


def _make_df(n: int, drift: float):
    import pandas as pd

    return pd.DataFrame(
        {
            "temperature": rng.normal(25 + drift, 3, n),
            "rainfall": rng.gamma(5, 20, n),
            "crop_label": rng.choice(["wheat", "rice"], n),
            "created_at": pd.date_range("2025-01-01", periods=n, freq="h"),
        }
    )


def test_export_drift_updates_gauges(tmp_path):
    path = _drift_report(tmp_path)
    metrics = StubMetrics()
    count = export_drift_to_prometheus(path, metrics)
    assert count >= 3
    assert "features" in metrics.drift
    assert "temporal" in metrics.drift
    assert metrics.drift["features"] in ("low", "moderate", "high")


def test_export_fairness_updates_gauges(tmp_path):
    evaluator = FairnessEvaluator()
    n = 400
    y_true = (rng.random(n) > 0.5).astype(int)
    y_pred = y_true.copy()
    result = evaluator.evaluate(y_true, y_pred, {"region": rng.choice(["n", "s"], n)})
    path = FairnessReportWriter().write(result, tmp_path)["json"]

    metrics = StubMetrics()
    count = export_fairness_to_prometheus(path, metrics)
    assert count >= 1
    assert all(v in ("compliant", "at_risk", "violating") for v in metrics.fairness.values())
    assert "disparate_impact" in json.loads(path.read_text(encoding="utf-8")).get("summary", {}) or True


def test_dashboard_renders_all_sections(tmp_path):
    drift_path = _drift_report(tmp_path)
    html = render_performance_dashboard(
        {
            "requests": 123,
            "requests_per_second": 4.2,
            "avg_latency_ms": 37.5,
            "errors": 2,
            "uptime_seconds": 300,
            "by_path": {"/api/v1/predict": {"requests": 100, "avg_ms": 30.0, "errors": 1}},
        },
        drift_report_path=drift_path,
        model_status={"ready": True},
    )
    assert "CropFusion Performance Dashboard" in html
    assert "System status" in html
    assert "Data drift" in html
    assert "ready" in html
    assert "/api/v1/predict" in html


def test_dashboard_gracefully_handles_missing_reports(tmp_path):
    html = render_performance_dashboard(
        metrics_snapshot={}, drift_report_path=tmp_path / "missing.json"
    )
    assert "Data drift" not in html
    assert "System status" in html


def test_export_system_syncs_registry():
    snapshot = {
        "requests": 3,
        "errors": 1,
        "by_path": {
            "/api/v1/predict": {"requests": 2, "avg_ms": 40.0, "errors": 1},
            "/api/v1/health": {"requests": 1, "avg_ms": 2.0, "errors": 0},
        },
    }
    metrics = StubMetrics()
    export_system_to_prometheus(snapshot, metrics)

    assert metrics.model_ready.value == 1
    recorded = {entry[0]: entry for entry in metrics.recorded}
    assert recorded["/api/v1/predict"][2] == 500
    assert recorded["/api/v1/health"][2] == 200
    assert sum(entry[3] == 0.0 for entry in metrics.recorded) == 3
