"""Low-level distribution statistics used by every drift analyzer.

All functions are pure numpy/scipy and operate on ``np.ndarray`` inputs, so
they are trivially testable in isolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _safe_series(values: Any) -> pd.Series:
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    return series.dropna().reset_index(drop=True)


def _histograms(
    reference: Any,
    current: Any,
    *,
    bins: int,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram reference and current using shared (combined) edges.

    Edges are derived from the pooled distribution so that the divergence
    between ``(a, b)`` and ``(b, a)`` is identical (bins are symmetric).
    """
    ref = _safe_series(reference).astype("float64")
    cur = _safe_series(current).astype("float64")
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("empty series cannot be binned")
    pooled = np.concatenate([ref.to_numpy(), cur.to_numpy()])
    edges = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1)))
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_hist = ref_hist.astype("float64")
    cur_hist = cur_hist.astype("float64")
    ref_prob = ref_hist / ref_hist.sum()
    cur_prob = cur_hist / cur_hist.sum()
    ref_prob = np.clip(ref_prob, clip, None)
    cur_prob = np.clip(cur_prob, clip, None)
    return ref_prob, cur_prob


def psi(reference: Any, current: Any, *, bins: int = 10, clip: float = 1e-4) -> float:
    """Population Stability Index between two numeric distributions."""
    ref_prob, cur_prob = _histograms(reference, current, bins=bins, clip=clip)
    delta = cur_prob - ref_prob
    return float(np.sum(delta * np.log(cur_prob / ref_prob)))


def kl_divergence(
    reference: Any, current: Any, *, bins: int = 10, clip: float = 1e-4
) -> float:
    """Kullback–Leibler divergence D_KL(current || reference)."""
    ref_prob, cur_prob = _histograms(reference, current, bins=bins, clip=clip)
    return float(np.sum(cur_prob * np.log(cur_prob / ref_prob)))


def js_divergence(
    reference: Any, current: Any, *, bins: int = 10, clip: float = 1e-4
) -> float:
    """Jensen–Shannon divergence (symmetric, bounded [0, ln2])."""
    ref_prob, cur_prob = _histograms(reference, current, bins=bins, clip=clip)
    midpoint = 0.5 * (ref_prob + cur_prob)
    left = np.sum(ref_prob * np.log(ref_prob / midpoint))
    right = np.sum(cur_prob * np.log(cur_prob / midpoint))
    return float(0.5 * (left + right))


def ks_test(reference: Any, current: Any) -> dict[str, float]:
    """Two-sample Kolmogorov–Smirnov test on raw values."""
    ref = _safe_series(reference).astype("float64").to_numpy()
    cur = _safe_series(current).astype("float64").to_numpy()
    stat, p_value = stats.ks_2samp(ref, cur)
    return {"statistic": float(stat), "p_value": float(p_value)}


def wasserstein_distance(reference: Any, current: Any) -> float:
    """Earth-mover's distance between two 1-D distributions."""
    ref = _safe_series(reference).astype("float64").to_numpy()
    cur = _safe_series(current).astype("float64").to_numpy()
    return float(stats.wasserstein_distance(ref, cur))


def chi2_test(reference: Any, current: Any) -> dict[str, float]:
    """Chi-squared test of independence for two aligned category count arrays.

    Uses the contingency table ``[[reference], [current]]`` so the two
    distributions may have different total sample sizes.

    Args:
        reference / current: aligned absolute counts (one entry per category).
    """
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    if ref.shape != cur.shape:
        raise ValueError("reference and current counts must be aligned")
    if ref.sum() == 0 or cur.sum() == 0:
        raise ValueError("empty count arrays cannot be compared")
    table = np.vstack([ref, cur])
    stat, p_value, _, _ = stats.chi2_contingency(table)
    return {"statistic": float(stat), "p_value": float(p_value)}


def category_shares(labels: Any) -> pd.Series:
    """Relative frequency of each category (index = category)."""
    series = pd.Series(labels).dropna()
    counts = series.value_counts(normalize=True)
    return counts


def categorical_drift(
    reference: Any,
    current: Any,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Categorical distribution drift: chi², JS, and per-category shifts."""
    ref_counts = pd.Series(reference).dropna().value_counts()
    cur_counts = pd.Series(current).dropna().value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    if not categories:
        raise ValueError("no categories present")

    ref_full = np.asarray([float(ref_counts.get(c, 0.0)) for c in categories])
    cur_full = np.asarray([float(cur_counts.get(c, 0.0)) for c in categories])
    test = chi2_test(ref_full, cur_full)

    ref_share = ref_full / ref_full.sum()
    cur_share = cur_full / cur_full.sum()
    mid = 0.5 * (ref_share + cur_share)
    js = float(
        0.5
        * (
            np.sum(ref_share * np.log(np.clip(ref_share / mid, 1e-12, None)))
            + np.sum(cur_share * np.log(np.clip(cur_share / mid, 1e-12, None)))
        )
    )

    shifts = [
        {
            "category": c,
            "reference_share": float(r),
            "current_share": float(q),
            "absolute_shift": float(q - r),
            "relative_change": float((q - r) / max(r, 1e-9)),
        }
        for c, r, q in zip(categories, ref_share, cur_share)
    ]
    shifts.sort(key=lambda row: abs(row["absolute_shift"]), reverse=True)

    return {
        "chi2": test,
        "js": js,
        "drifted": bool(test["p_value"] < alpha or js > 0.05),
        "new_categories": sorted(set(cur_counts.index) - set(ref_counts.index)),
        "vanished_categories": sorted(set(ref_counts.index) - set(cur_counts.index)),
        "category_shifts": shifts,
        "num_categories": len(categories),
    }


def confidence_to_probability(confidences: Any, classes: int) -> np.ndarray:
    """Convert a scalar-confidence vector to a soft class-probability matrix.

    Confidence ``c`` is spread uniformly across the other ``classes - 1``
    outputs so entropy and divergence metrics behave sensibly.
    """
    conf = np.asarray(_safe_series(confidences), dtype="float64")
    n = len(conf)
    matrix = np.zeros((n, max(classes, 1)), dtype="float64")
    for i, c in enumerate(conf):
        c = float(np.clip(c, 0.0, 1.0))
        share = (1.0 - c) / max(classes - 1, 1) if classes > 1 else 0.0
        matrix[i, 0] = c
        if classes > 1:
            matrix[i, 1:] = share
    return matrix
