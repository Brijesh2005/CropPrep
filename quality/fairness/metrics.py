"""Group-level classification, regression and calibration metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Standard classification metrics for a single group."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {}
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    base_rate = float(y_true.mean())

    def _safe(denominator: float, numerator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "support": int(n),
        "base_rate": base_rate,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tpr": _safe(tp + fn, tp),
        "fpr": _safe(fp + tn, fp),
        "tnr": _safe(tp + fn, tn),
        "fnr": _safe(fp + tn, fn),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """MAE / RMSE for a continuous outcome group."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if len(y_true) == 0:
        return {}
    return {
        "support": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_true": float(y_true.mean()),
        "mean_pred": float(y_pred.mean()),
        "signed_bias": float(y_pred.mean() - y_true.mean()),
    }


def expected_calibration_error(
    y_true: Sequence[int], y_proba: Sequence[float], *, bins: int = 10
) -> dict[str, Any]:
    """Expected Calibration Error plus the per-bin calibration curve."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype="float64")
    if len(y_true) == 0:
        return {"ece": 0.0, "curve": [], "confidence": 0.0}
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    curve: list[dict[str, float]] = []
    total = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if lo == 0.0:
            mask |= y_proba == 0.0
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_proba[mask].mean())
        ece += count / total * abs(acc - conf)
        curve.append({"bin_low": lo, "bin_high": hi, "count": count, "accuracy": acc, "confidence": conf})
    return {"ece": ece, "curve": curve, "confidence": float(y_proba.mean())}


def roc_auc(y_true: Sequence[int], y_proba: Sequence[float]) -> float:
    """ROC-AUC (positive class) for a group."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype="float64")
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_proba))
