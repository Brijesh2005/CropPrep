"""Temporal drift analyzer — drift evolution over time.

Splits the current dataset into time windows (monthly by default) and scores
each window against the fixed reference. Also detects monotonic drift trends
via linear regression on the per-window drift metric and counts "episodes"
(windows exceeding the alert threshold).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import DriftConfig
from .result import TemporalDriftResult
from .statistical import psi


class TemporalDriftAnalyzer:
    """Track how a drift metric evolves across consecutive time windows."""

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    def analyze(
        self,
        reference: pd.Series,
        current: pd.Series,
        timestamps: Sequence[Any],
        *,
        metric: str = "psi",
    ) -> TemporalDriftResult:
        """Analyse drift over time for a single numeric series.

        Args:
            reference: Reference distribution (static baseline).
            current: Current values.
            timestamps: One timestamp per current value (same length).
            metric: Drift metric computed per window (``psi`` supported).
        """
        cfg = self.config
        frame = pd.DataFrame({"value": pd.Series(current).to_numpy(), "ts": pd.to_datetime(timestamps)})
        frame = frame.dropna(subset=["value"]).sort_values("ts")
        if frame.empty:
            raise ValueError("no dated current values")

        windows: list[dict[str, Any]] = []
        for ts, group in frame.groupby(pd.Grouper(key="ts", freq=cfg.temporal_window)):
            values = group["value"].dropna()
            if len(values) < 2:
                continue
            drift_value = float(psi(reference, values, bins=cfg.bins, clip=cfg.clip))
            severity = cfg.severity_from_psi(drift_value)
            windows.append(
                {
                    "window": str(pd.Timestamp(ts).date()) if not pd.isna(ts) else "unknown",
                    "samples": int(len(values)),
                    "metric": metric,
                    "value": drift_value,
                    "severity": severity,
                    "drifted": severity != "low",
                }
            )

        if not windows:
            return TemporalDriftResult(
                dimension="temporal", severity="low", drifted=False, windows=[],
                latest_severity="low",
            )

        series = np.asarray([w["value"] for w in windows], dtype="float64")
        indices = np.arange(len(series), dtype="float64")
        slope, intercept, r_value, p_value, _ = stats.linregress(indices, series)

        trend_direction = "stable"
        if p_value < cfg.temporal_trend_alpha:
            trend_direction = "increasing" if slope > 0 else "decreasing"

        episode_count = int(sum(1 for w in windows if w["drifted"]))
        latest_severity = windows[-1]["severity"]
        severity = max((w["severity"] for w in windows), key=_rank)

        drifted = bool(episode_count > 0 or (trend_direction == "increasing" and p_value < cfg.temporal_trend_alpha))
        if severity == "low" and drifted:
            severity = "moderate"

        return TemporalDriftResult(
            dimension="temporal",
            severity=severity,
            drifted=drifted,
            windows=windows,
            trend_slope=float(slope),
            trend_p_value=float(p_value),
            trend_direction=trend_direction,
            episode_count=episode_count,
            latest_severity=latest_severity,
            metrics={
                "num_windows": len(windows),
                "latest_value": float(series[-1]),
                "mean_value": float(series.mean()),
                "max_value": float(series.max()),
            },
        )


def _rank(severity: str) -> int:
    return {"low": 0, "moderate": 1, "high": 2}.get(severity, 0)
