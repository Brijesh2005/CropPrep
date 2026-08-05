"""Fairness framework test-suite."""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.quality.fairness import FairnessConfig, FairnessEvaluator, FairnessReportWriter, RegionalFairnessEvaluator
from training.quality.fairness.metrics import (
    classification_metrics,
    expected_calibration_error,
    regression_metrics,
    roc_auc,
)

rng = np.random.default_rng(3)


def test_classification_metrics_values():
    y_true = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    m = classification_metrics(y_true, y_pred)
    assert m["support"] == 10
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["tpr"] == pytest.approx(0.8333, abs=1e-3)
    assert m["fpr"] == pytest.approx(0.25, abs=1e-3)


def test_regression_metrics():
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 290.0]
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(10.0)
    assert m["signed_bias"] == pytest.approx(-10.0 / 3, abs=1e-9)


def test_calibration_ece_perfect_is_zero():
    # Perfect calibration: confidence == accuracy in every bin.
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_proba = np.full(len(y_true), 0.6)
    result = expected_calibration_error(y_true, y_proba, bins=10)
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)


def test_roc_auc_perfect():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.9, 0.8]
    assert roc_auc(y_true, y_proba) == pytest.approx(1.0)


def test_classification_metrics_empty_is_empty():
    assert classification_metrics([], []) == {}


def test_roc_auc_single_class_is_zero():
    assert roc_auc([0, 0, 0], [0.9, 0.8, 0.7]) == 0.0


def test_ece_empty_is_zero():
    result = expected_calibration_error([], [])
    assert result["ece"] == 0.0
    assert result["confidence"] == 0.0


def test_fairness_compliant_when_groups_balanced():
    evaluator = FairnessEvaluator()
    n = 600
    y_true = (rng.random(n) > 0.5).astype(int)
    y_pred = y_true.copy()
    group = rng.choice(["a", "b"], n)

    result = evaluator.evaluate(y_true, y_pred, {"region": group})
    assert result.overall_status == "compliant"
    assert all(v.status == "compliant" for v in result.verdicts)


def test_fairness_violating_disparate_impact():
    evaluator = FairnessEvaluator()
    n_a, n_b = 500, 500
    y_true = np.array([1] * n_a + [0] * n_b)
    y_pred = y_true.copy()
    group = np.array(["privileged"] * n_a + ["underprivileged"] * n_b)

    result = evaluator.evaluate(y_true, y_pred, {"group": group})
    verdict = next(v for v in result.verdicts if v.metric == "disparate_impact")
    assert verdict.value == pytest.approx(0.0)
    assert verdict.status == "violating"


def test_fairness_at_risk_thresholds():
    cfg = FairnessConfig()
    evaluator = FairnessEvaluator(cfg)
    n = 500
    y_true = np.array([1] * 250 + [0] * 250)
    y_pred = y_true.copy()
    group = np.array(["a"] * 250 + ["b"] * 250)
    result = evaluator.evaluate(y_true, y_pred, {"g": group})
    assert result.overall_status in ("compliant", "at_risk", "violating")
    assert len(result.groups) == 2


def test_fairness_insufficient_group_flagged():
    evaluator = FairnessEvaluator(FairnessConfig(min_group_size=100))
    y_true = [1] * 5 + [0] * 200
    y_pred = [1] * 5 + [0] * 200
    group = ["tiny"] * 5 + ["large"] * 200
    result = evaluator.evaluate(y_true, y_pred, {"size": group})
    assert any(g.metrics.get("status") == "insufficient_data" for g in result.groups)


def test_fairness_with_probabilities():
    evaluator = FairnessEvaluator()
    n = 400
    y_true = (rng.random(n) > 0.5).astype(int)
    y_pred = y_true.copy()
    y_proba = np.clip(y_true.astype(float) * 0.9 + rng.random(n) * 0.1, 0, 1)
    group = rng.choice(["clay", "sand"], n)
    result = evaluator.evaluate(y_true, y_pred, {"soil": group}, y_proba=y_proba)
    assert result.verdicts
    assert any(v.metric == "calibration_parity" for v in result.verdicts)


def test_regional_fairness_report():
    evaluator = RegionalFairnessEvaluator()
    regions = ["Punjab", "Punjab", "Gujarat", "Gujarat", "Bihar", "Bihar", "Kerala", "Kerala"] * 60
    n = len(regions)
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 60)
    y_pred = y_true.copy()
    report = evaluator.evaluate(
        y_true, y_pred, regions,
        region_centroids={
            "Punjab": (75.0, 30.5), "Gujarat": (72.0, 22.5),
            "Bihar": (85.5, 25.5), "Kerala": (76.5, 10.0),
        },
    )
    assert report["attribute"] == "region"
    assert len(report["regions"]) == 4
    assert report["regions"][0]["centroid_lon"] is not None


def test_fairness_regression_verdicts():
    evaluator = FairnessEvaluator()
    n = 200
    y_true = rng.uniform(100, 200, 2 * n)
    y_pred = y_true.copy()
    y_pred[:n] += 50  # one group systematically over-predicts
    group = ["a"] * n + ["b"] * n

    result = evaluator.evaluate(
        y_true, y_pred, {"region": group}, y_pred_regression=y_pred, task="regression"
    )
    assert result.task == "regression"
    metrics = {v.metric: v for v in result.verdicts}
    assert "error_parity" in metrics
    assert "signed_bias_parity" in metrics
    assert metrics["signed_bias_parity"].status == "violating"


def test_fairness_aggregate_without_attributes():
    result = FairnessEvaluator().evaluate([], [], {})
    assert result.overall_status == "compliant"
    assert result.attribute == "none"
    assert result.verdicts == []


def test_fairness_result_serialisable():
    evaluator = FairnessEvaluator()
    n = 200
    y_true = (rng.random(n) > 0.5).astype(int)
    result = evaluator.evaluate(y_true, y_true.copy(), {"g": rng.choice(["x", "y"], n)})
    data = result.to_dict()
    json.dumps(data)
    assert data["task"] == "classification"
    assert data["attribute"] == "g"
    assert data["groups"][0]["support"] > 0
    assert data["verdicts"][0]["status"] in ("compliant", "at_risk", "violating")
    assert data["summary"]["attributes"] == ["g"]


def test_report_writer_all_formats(tmp_path):
    evaluator = FairnessEvaluator()
    n = 400
    y_true = (rng.random(n) > 0.4).astype(int)
    y_pred = y_true.copy()
    group = rng.choice(["north", "south"], n)
    result = evaluator.evaluate(y_true, y_pred, {"zone": group})
    paths = FairnessReportWriter().write(result, tmp_path)
    assert set(paths) == {"json", "csv", "html"}
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["overall_status"] in ("compliant", "at_risk", "violating")
    assert len(data["groups"]) == 2
