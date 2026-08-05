"""Feature-level distribution drift analyzer."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .config import DriftConfig
from .exceptions import InsufficientDataError
from .result import FeatureDriftResult
from .statistical import categorical_drift, js_divergence, ks_test, psi, wasserstein_distance


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


class FeatureDriftAnalyzer:
    """Compare reference vs current feature distributions.

    Numeric features are scored with PSI, JS, KS and Wasserstein; categorical
    features with chi² and JS on category shares.
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    def analyze(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
    ) -> list[FeatureDriftResult]:
        """Return one :class:`FeatureDriftResult` per analysed column."""
        if len(reference) < self.config.min_samples:
            raise InsufficientDataError(
                f"reference needs >= {self.config.min_samples} samples, got {len(reference)}"
            )
        if len(current) < self.config.min_samples:
            raise InsufficientDataError(
                f"current needs >= {self.config.min_samples} samples, got {len(current)}"
            )

        names = list(columns) if columns is not None else list(reference.columns)
        results: list[FeatureDriftResult] = []
        for name in names:
            if name not in reference.columns or name not in current.columns:
                continue
            ref = reference[name].dropna()
            cur = current[name].dropna()
            if len(ref) < 2 or len(cur) < 2:
                continue
            if _is_numeric(ref) and _is_numeric(cur):
                results.append(self._numeric(name, ref, cur))
            else:
                results.append(self._categorical(name, ref, cur))
        return results

    # ------------------------------------------------------------------ #

    def _numeric(
        self, name: str, reference: pd.Series, current: pd.Series
    ) -> FeatureDriftResult:
        cfg = self.config
        psi_value = psi(reference, current, bins=cfg.bins, clip=cfg.clip)
        js_value = js_divergence(reference, current, bins=cfg.bins, clip=cfg.clip)
        ks = ks_test(reference, current)
        wd = wasserstein_distance(reference, current)

        drifted = ks["p_value"] < cfg.ks_alpha
        severity = cfg.severity_from_psi(psi_value)
        if severity != "high":
            severity = max(severity, cfg.severity_from_js(js_value), key=_severity_rank)

        return FeatureDriftResult(
            dimension="feature",
            feature=name,
            dtype="numeric",
            severity=severity,
            drifted=drifted,
            metrics={
                "psi": psi_value,
                "js": js_value,
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "wasserstein": wd,
                "reference_mean": float(reference.mean()),
                "current_mean": float(current.mean()),
                "reference_std": float(reference.std()),
                "current_std": float(current.std()),
            },
        )

    def _categorical(
        self, name: str, reference: pd.Series, current: pd.Series
    ) -> FeatureDriftResult:
        cfg = self.config
        result = categorical_drift(reference, current, alpha=cfg.chi2_alpha)
        js_value = result["js"]
        severity = cfg.severity_from_js(js_value)
        return FeatureDriftResult(
            dimension="feature",
            feature=name,
            dtype="categorical",
            severity=severity,
            drifted=result["drifted"],
            metrics={
                "chi2_statistic": result["chi2"]["statistic"],
                "chi2_p_value": result["chi2"]["p_value"],
                "js": js_value,
                "new_categories": result["new_categories"],
                "vanished_categories": result["vanished_categories"],
            },
            details={"category_shifts": result["category_shifts"][: cfg.bins]},
        )


def _severity_rank(severity: str) -> int:
    return {"low": 0, "moderate": 1, "high": 2}.get(severity, 0)
