"""Label / outcome distribution drift analyzer.

Supports both categorical outcomes (crop classes — the CropFusion primary
label) and continuous outcomes (yield, kg/ha). Detects novelty (classes the
model never saw), vanishing classes, and statistical distribution shift.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import DriftConfig
from .exceptions import InsufficientDataError
from .result import LabelDriftResult
from .statistical import categorical_drift, js_divergence, ks_test, wasserstein_distance


class LabelDriftAnalyzer:
    """Measure drift of the ground-truth label distribution."""

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    def analyze(
        self,
        reference: Any,
        current: Any,
        *,
        task: str = "classification",
    ) -> LabelDriftResult:
        """Compare reference and current label distributions."""
        if len(pd.Series(reference).dropna()) < self.config.min_samples:
            raise InsufficientDataError("reference labels are too small")
        if len(pd.Series(current).dropna()) < self.config.min_samples:
            raise InsufficientDataError("current labels are too small")

        if task == "regression":
            return self._regression(reference, current)
        return self._classification(reference, current)

    # ------------------------------------------------------------------ #

    def _classification(
        self, reference: Any, current: Any
    ) -> LabelDriftResult:
        cfg = self.config
        result = categorical_drift(reference, current, alpha=cfg.label_alpha)
        severity = cfg.severity_from_js(result["js"])
        return LabelDriftResult(
            dimension="label",
            task="classification",
            severity=severity,
            drifted=result["drifted"],
            metrics={
                "js": result["js"],
                "chi2_statistic": result["chi2"]["statistic"],
                "chi2_p_value": result["chi2"]["p_value"],
            },
            categories=result["category_shifts"],
            novelty=result["new_categories"],
            vanished=result["vanished_categories"],
        )

    def _regression(self, reference: Any, current: Any) -> LabelDriftResult:
        cfg = self.config
        ref = pd.Series(reference).dropna()
        cur = pd.Series(current).dropna()
        js_value = js_divergence(ref, cur, bins=cfg.bins, clip=cfg.clip)
        ks = ks_test(ref, cur)
        wd = wasserstein_distance(ref, cur)

        drifted = ks["p_value"] < cfg.ks_alpha
        severity = cfg.severity_from_js(js_value)
        return LabelDriftResult(
            dimension="label",
            task="regression",
            severity=severity,
            drifted=drifted,
            metrics={
                "js": js_value,
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "wasserstein": wd,
                "reference_mean": float(ref.mean()),
                "current_mean": float(cur.mean()),
                "reference_std": float(ref.std()),
                "current_std": float(cur.std()),
                "mean_shift": float(cur.mean() - ref.mean()),
            },
        )
