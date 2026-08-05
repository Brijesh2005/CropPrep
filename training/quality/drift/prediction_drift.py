"""Prediction drift analyzer — monitors model output distributions in prod.

Two supported modes:

* ``classification`` — ``current`` is a matrix of per-class probabilities
  (``[N, K]``) and ``reference`` the equivalent training-time matrix. Drift is
  scored on the mean probability vector (JS), the argmax class distribution
  (chi²/JS), and the shift in confidence and prediction entropy.
* ``regression`` — scalar predictions (yield) scored with KS / JS / Wasserstein.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DriftConfig
from .result import PredictionDriftResult
from .statistical import (
    categorical_drift,
    confidence_to_probability,
    js_divergence,
    ks_test,
    wasserstein_distance,
)


class PredictionDriftAnalyzer:
    """Measure drift of model outputs between reference and current batches."""

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    def analyze(
        self,
        reference: Any,
        current: Any,
        *,
        mode: str = "classification",
    ) -> PredictionDriftResult:
        """Score prediction drift.

        Args:
            reference: ``[N, K]`` probability matrix or 1-D scalar predictions.
            current: Same shape as ``reference``.
            mode: ``"classification"`` or ``"regression"``.
        """
        if mode == "regression":
            return self._regression(reference, current)
        return self._classification(reference, current)

    # ------------------------------------------------------------------ #

    def _classification(
        self, reference: Any, current: Any
    ) -> PredictionDriftResult:
        cfg = self.config
        ref = np.asarray(reference, dtype="float64")
        cur = np.asarray(current, dtype="float64")

        if ref.ndim == 1:
            ref = confidence_to_probability(ref, classes=3)
        if cur.ndim == 1:
            cur = confidence_to_probability(cur, classes=3)
        if ref.shape[1] != cur.shape[1]:
            raise ValueError("reference and current class dimensions differ")

        mean_ref = ref.mean(axis=0)
        mean_cur = cur.mean(axis=0)
        js_value = _js_of_vectors(mean_ref, mean_cur)

        ref_pred = ref.argmax(axis=1)
        cur_pred = cur.argmax(axis=1)
        class_result = categorical_drift(ref_pred, cur_pred, alpha=cfg.chi2_alpha)

        ref_entropy = _entropy_bits(ref)
        cur_entropy = _entropy_bits(cur)
        entropy_shift = float(cur_entropy - ref_entropy)

        ref_conf = ref.max(axis=1)
        cur_conf = cur.max(axis=1)
        confidence_shift = float(cur_conf.mean() - ref_conf.mean())

        drifted = bool(
            class_result["drifted"]
            or abs(entropy_shift) > cfg.entropy_threshold
        )
        severity = cfg.severity_from_js(js_value)
        if abs(entropy_shift) > 2 * cfg.entropy_threshold and severity != "high":
            severity = "moderate"

        return PredictionDriftResult(
            dimension="prediction",
            mode="classification",
            severity=severity,
            drifted=drifted,
            confidence_shift=confidence_shift,
            entropy_shift=entropy_shift,
            metrics={
                "js": js_value,
                "mean_probability_reference": mean_ref.tolist(),
                "mean_probability_current": mean_cur.tolist(),
                "reference_confidence": float(ref_conf.mean()),
                "current_confidence": float(cur_conf.mean()),
                "reference_entropy_bits": float(ref_entropy),
                "current_entropy_bits": float(cur_entropy),
                "argmax_chi2_statistic": class_result["chi2"]["statistic"],
                "argmax_chi2_p_value": class_result["chi2"]["p_value"],
            },
            top_class_shifts=class_result["category_shifts"][:10],
        )

    def _regression(self, reference: Any, current: Any) -> PredictionDriftResult:
        cfg = self.config
        ref = pd.Series(reference).dropna()
        cur = pd.Series(current).dropna()
        js_value = js_divergence(ref, cur, bins=cfg.bins, clip=cfg.clip)
        ks = ks_test(ref, cur)
        wd = wasserstein_distance(ref, cur)
        drifted = ks["p_value"] < cfg.ks_alpha
        return PredictionDriftResult(
            dimension="prediction",
            mode="regression",
            severity=cfg.severity_from_js(js_value),
            drifted=drifted,
            metrics={
                "js": js_value,
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "wasserstein": wd,
                "reference_mean": float(ref.mean()),
                "current_mean": float(cur.mean()),
            },
        )


def _js_of_vectors(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    m = 0.5 * (p + q)
    left = float(np.sum(p * np.log(p / m)))
    right = float(np.sum(q * np.log(q / m)))
    return float(0.5 * (left + right))


def _entropy_bits(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, None)
    return float(-np.sum(p * np.log2(p)) / len(p))
