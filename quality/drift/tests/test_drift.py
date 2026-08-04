"""Drift framework test-suite."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quality.drift import DriftConfig, DriftMonitor, ReportWriter
from quality.drift.feature_drift import FeatureDriftAnalyzer
from quality.drift.label_drift import LabelDriftAnalyzer
from quality.drift.prediction_drift import PredictionDriftAnalyzer
from quality.drift.result import (
    DriftReport,
    DriftResult,
    FeatureDriftResult,
    rank_severity,
)
from quality.drift.spatial_drift import SpatialDriftAnalyzer
from quality.drift.statistical import (
    categorical_drift,
    chi2_test,
    js_divergence,
    kl_divergence,
    ks_test,
    psi,
    wasserstein_distance,
)
from quality.drift.temporal_drift import TemporalDriftAnalyzer

rng = np.random.default_rng(42)


def test_psi_identical_distributions_is_zero():
    values = rng.normal(0, 1, 2000)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distribution_is_positive_and_classifies():
    base = rng.normal(0, 1, 2000)
    shifted = rng.normal(2.0, 1.0, 2000)
    value = psi(base, shifted)
    assert value > 0.1
    cfg = DriftConfig()
    assert cfg.severity_from_psi(value) == "high"


def test_psi_never_negative_for_close_distributions():
    base = rng.normal(0, 1, 2000)
    for scale in (0.5, 1.5):
        assert psi(base, rng.normal(0, scale, 2000)) >= -1e-6


def test_config_severity_from_js_thresholds():
    cfg = DriftConfig()
    assert cfg.severity_from_js(0.02) == "low"
    assert cfg.severity_from_js(0.07) == "moderate"
    assert cfg.severity_from_js(0.15) == "high"


def test_config_severity_from_psi_boundaries():
    cfg = DriftConfig()
    assert cfg.severity_from_psi(cfg.psi_thresholds[0] - 1e-6) == "low"
    assert cfg.severity_from_psi(cfg.psi_thresholds[0]) == "moderate"
    assert cfg.severity_from_psi(cfg.psi_thresholds[1]) == "high"


def test_rank_severity_ordering():
    assert rank_severity("low") < rank_severity("moderate") < rank_severity("high")
    assert rank_severity("unknown") == rank_severity("low")


def test_drift_result_rejects_invalid_severity():
    with pytest.raises(ValueError):
        DriftResult(dimension="features", severity="critical")


def test_drift_report_serialisable_with_numpy_metrics():
    report = DriftReport(
        reference_samples=100,
        current_samples=80,
        overall_severity="high",
        drifted=True,
        features=[
            FeatureDriftResult(
                dimension="features",
                feature="temperature",
                severity="high",
                drifted=True,
                metrics={
                    "psi": np.float64(0.31),
                    "js": 0.12,
                    "ks_p_value": np.float32(0.001),
                },
            )
        ],
    )
    summary = report.summary()
    assert summary["dimensions"]["features"]["high"] == 1
    assert summary["dimensions"]["features"]["drifted"] == 1
    data = report.to_dict()
    json.dumps(data)  # numpy scalars must be converted so the report is JSON-serialisable
    assert data["features"][0]["metrics"]["psi"] == 0.31
    assert data["features"][0]["metrics"]["ks_p_value"] == pytest.approx(0.001)


def test_kl_and_js_nonnegative_and_symmetric_js():
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0.5, 1.2, 2000)
    assert kl_divergence(a, b) >= 0
    js_ab = js_divergence(a, b)
    js_ba = js_divergence(b, a)
    assert js_ab == pytest.approx(js_ba, abs=1e-6)
    assert 0 <= js_ab <= np.log(2) + 1e-9


def test_ks_test_pvalues():
    same = rng.normal(0, 1, 500)
    assert ks_test(same, same)["p_value"] > 0.05
    other = rng.normal(3, 1, 500)
    assert ks_test(same, other)["p_value"] < 0.01


def test_wasserstein_zero_for_identical():
    a = rng.normal(0, 1, 500)
    assert wasserstein_distance(a, a) == 0.0


def test_chi2_detects_shift():
    ref = np.array([100, 100, 100])
    cur_same = np.array([98, 102, 100])
    cur_shifted = np.array([250, 30, 20])
    assert chi2_test(ref, cur_same)["p_value"] > 0.05
    assert chi2_test(ref, cur_shifted)["p_value"] < 0.001


def test_categorical_drift_identical():
    result = categorical_drift(
        ["a"] * 300 + ["b"] * 200, ["a"] * 300 + ["b"] * 200, alpha=0.05
    )
    assert result["drifted"] is False
    assert result["new_categories"] == []


def test_categorical_drift_novelty():
    result = categorical_drift(["a"] * 400 + ["b"] * 100, ["a"] * 100 + ["c"] * 400)
    assert result["new_categories"] == ["c"]
    assert result["vanished_categories"] == ["b"]


def test_feature_analyzer_identical_data_low_severity():
    reference = pd.DataFrame({"temperature": rng.normal(25, 3, 1000), "zone": "north"})
    current = pd.DataFrame({"temperature": rng.normal(25, 3, 1000), "zone": "north"})
    results = FeatureDriftAnalyzer().analyze(reference, current)
    assert results
    assert all(r.severity == "low" for r in results)
    assert all(not r.drifted for r in results)


def test_feature_analyzer_detects_numeric_shift():
    reference = pd.DataFrame({"temperature": rng.normal(25, 3, 2000)})
    current = pd.DataFrame({"temperature": rng.normal(35, 3, 2000)})
    results = FeatureDriftAnalyzer().analyze(reference, current)
    numeric = next(r for r in results if r.feature == "temperature")
    assert numeric.severity == "high"
    assert numeric.metrics["psi"] > 0.25


def test_feature_analyzer_detects_categorical_shift():
    reference = pd.DataFrame({"soil_type": ["clay"] * 400 + ["sand"] * 100})
    current = pd.DataFrame({"soil_type": ["sand"] * 400 + ["clay"] * 100})
    results = FeatureDriftAnalyzer().analyze(reference, current)
    categorical = next(r for r in results if r.feature == "soil_type")
    assert categorical.severity != "low"


def test_label_analyzer_classification():
    analyzer = LabelDriftAnalyzer()
    reference = ["wheat"] * 300 + ["rice"] * 300 + ["maize"] * 200
    current = ["rice"] * 500 + ["maize"] * 300
    result = analyzer.analyze(reference, current)
    assert result.dimension == "label"
    assert result.task == "classification"
    assert result.drifted is True


def test_label_analyzer_regression_mean_shift():
    analyzer = LabelDriftAnalyzer()
    reference = rng.normal(4000, 500, 1000)
    current = rng.normal(3200, 500, 1000)
    result = analyzer.analyze(reference, current, task="regression")
    assert result.task == "regression"
    assert result.drifted is True
    assert result.metrics["mean_shift"] < 0


def test_prediction_drift_identical_probs():
    probs = np.random.default_rng(7).dirichlet(np.ones(5), size=500)
    result = PredictionDriftAnalyzer().analyze(probs, probs.copy())
    assert result.drifted is False
    assert result.metrics["js"] < 1e-6


def test_prediction_drift_detects_confidence_collapse():
    peaked = np.eye(4)[np.random.default_rng(1).integers(0, 4, 400)].astype(float)
    flat = np.full((400, 4), 0.25)
    result = PredictionDriftAnalyzer().analyze(peaked, flat)
    assert result.entropy_shift > 0
    assert result.drifted is True


def test_prediction_drift_regression():
    result = PredictionDriftAnalyzer().analyze(
        rng.normal(4000, 500, 800), rng.normal(3400, 500, 800), mode="regression"
    )
    assert result.drifted is True


def test_spatial_drift_same_region_low():
    reference = [(lon, lat) for lon in range(70, 80) for lat in range(10, 20) for _ in range(2)]
    current = [(lon + 0.1, lat + 0.1) for lon in range(70, 80) for lat in range(10, 20)]
    result = SpatialDriftAnalyzer().analyze(reference, current)
    assert result.severity == "low"
    assert result.mean_nearest_neighbour_km < 50


def test_spatial_drift_far_region_high():
    reference = [(lon, lat) for lon in range(70, 80) for lat in range(10, 20)]
    current = [(lon + 40, lat + 30) for lon in range(70, 80) for lat in range(10, 20)]
    result = SpatialDriftAnalyzer().analyze(reference, current)
    assert result.severity == "high"
    assert result.novel_cell_share > 0.5
    assert result.mean_nearest_neighbour_km > 1000


def test_temporal_drift_stable():
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(0, 1, 5000)
    timestamps = pd.date_range("2025-01-01", periods=5000, freq="h")
    result = TemporalDriftAnalyzer().analyze(reference, current, timestamps)
    assert result.trend_direction == "stable"
    assert result.episode_count == 0


def test_temporal_drift_increasing():
    reference = rng.normal(0, 1, 5000)
    drift = np.linspace(0, 3, 5000)
    current = rng.normal(drift, 1, 5000)
    timestamps = pd.date_range("2025-01-01", periods=5000, freq="h")
    result = TemporalDriftAnalyzer().analyze(reference, current, timestamps)
    assert result.episode_count > 0
    assert result.severity in ("moderate", "high")


def _reference_df(n: int = 1500) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": rng.normal(25, 3, n),
            "rainfall": rng.gamma(5, 20, n),
            "soil_type": rng.choice(["clay", "sand", "loam"], n, p=[0.5, 0.3, 0.2]),
            "crop_label": rng.choice(["wheat", "rice", "maize"], n, p=[0.4, 0.35, 0.25]),
            "created_at": pd.date_range("2025-01-01", periods=n, freq="h"),
        }
    )


def test_drift_monitor_full_report_drifted():
    reference = _reference_df()
    current = _reference_df(1200)
    current["temperature"] = rng.normal(38, 4, len(current))
    current["crop_label"] = rng.choice(["rice", "maize", "sorghum"], len(current), p=[0.6, 0.3, 0.1])
    current["created_at"] = pd.date_range("2025-04-01", periods=len(current), freq="h")

    monitor = DriftMonitor(
        reference,
        feature_columns=["temperature", "rainfall", "soil_type"],
        label_column="crop_label",
    )
    report = monitor.evaluate(current, timestamp_column="created_at")
    assert report.drifted is True
    assert report.overall_severity in ("moderate", "high")
    assert any(f.feature == "temperature" and f.severity == "high" for f in report.features)
    assert report.labels is not None
    assert report.labels.drifted is True


def test_drift_monitor_full_report_clean():
    reference = _reference_df(1500)
    current = reference.sample(1000, random_state=3)
    monitor = DriftMonitor(reference, feature_columns=["temperature", "rainfall", "soil_type"])
    report = monitor.evaluate(current)
    assert report.drifted is False
    assert report.overall_severity == "low"


def test_report_writer_all_formats(tmp_path):
    report = DriftMonitor(_reference_df(), feature_columns=["temperature"]).evaluate(
        _reference_df(500)
    )
    paths = ReportWriter().write(report, tmp_path)
    assert set(paths) == {"json", "csv", "html", "pdf"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["overall_severity"] == "low"
    assert paths["html"].read_text(encoding="utf-8").startswith("<!doctype html>")
