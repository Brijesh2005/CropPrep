"""Spatial coverage drift analyzer.

Quantifies whether production inference requests are arriving from regions the
model was trained/validated on. Uses a longitude/latitude grid (approximate
geohash) for occupancy comparison plus a KD-tree nearest-neighbour coverage
distance so "how far is this new request from any training location?" becomes
a concrete number.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import DriftConfig
from .result import SpatialDriftResult
from .statistical import categorical_drift


def grid_cell(lon: float, lat: float, cell_degrees: float) -> tuple[int, int]:
    """Quantise a lon/lat point to a grid cell key."""
    return (int(np.floor(lon / cell_degrees)), int(np.floor(lat / cell_degrees)))


class SpatialDriftAnalyzer:
    """Measure geographic coverage drift between reference and current data."""

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    def analyze(
        self,
        reference_lon_lat: Sequence[tuple[float, float]],
        current_lon_lat: Sequence[tuple[float, float]],
    ) -> SpatialDriftResult:
        """Score spatial drift from two ``[(lon, lat), ...]`` collections."""
        ref = np.asarray(reference_lon_lat, dtype="float64").reshape(-1, 2)
        cur = np.asarray(current_lon_lat, dtype="float64").reshape(-1, 2)
        if ref.shape[0] < 2 or cur.shape[0] < 2:
            raise ValueError("need at least two points per collection")

        cell = self.config.spatial_cell_degrees

        def occupancy(points: np.ndarray) -> pd.Series:
            return pd.Series(
                {f"{g[0]},{g[1]}": 1 for g in (grid_cell(lon, lat, cell) for lon, lat in points)},
                dtype="int64",
            ).groupby(level=0).sum()

        ref_cells = occupancy(ref)
        cur_cells = occupancy(cur)
        novel = sorted(set(cur_cells.index) - set(ref_cells.index))

        cat = categorical_drift(ref_cells, cur_cells, alpha=self.config.chi2_alpha)
        novel_share = float(len(novel) / max(len(cur_cells), 1))

        coverage = self._nearest_neighbour_km(ref, cur)

        hot_cells = self._extreme_cells(ref_cells, cur_cells, top=True)
        cold_cells = self._extreme_cells(ref_cells, cur_cells, top=False)

        drifted = bool(
            novel_share > self.config.spatial_max_novel_cells
            or cat["drifted"]
        )
        severity = "high" if novel_share > 2 * self.config.spatial_max_novel_cells else (
            "moderate" if drifted else "low"
        )

        return SpatialDriftResult(
            dimension="spatial",
            severity=severity,
            drifted=drifted,
            cell_size_degrees=float(cell),
            num_cells_reference=int(len(ref_cells)),
            num_cells_current=int(len(cur_cells)),
            novel_cell_share=novel_share,
            mean_nearest_neighbour_km=coverage,
            hot_cells=hot_cells,
            cold_cells=cold_cells,
            metrics={
                "js": cat["js"],
                "chi2_statistic": cat["chi2"]["statistic"],
                "chi2_p_value": cat["chi2"]["p_value"],
                "novel_cells": novel[:20],
                "reference_points": int(ref.shape[0]),
                "current_points": int(cur.shape[0]),
            },
        )

    @staticmethod
    def _nearest_neighbour_km(ref: np.ndarray, cur: np.ndarray) -> float:
        """Mean great-circle distance from each current point to the nearest ref point."""
        ref_rad = np.radians(ref)
        cur_rad = np.radians(cur)
        tree = cKDTree(ref_rad)
        distances, _ = tree.query(cur_rad, k=1)
        earth_km = 6371.0
        return float(np.mean(earth_km * distances))

    @staticmethod
    def _extreme_cells(
        ref: pd.Series, cur: pd.Series, *, top: bool
    ) -> list[dict[str, Any]]:
        all_cells = sorted(set(ref.index) | set(cur.index))
        rows = []
        for cell_id in all_cells:
            r = float(ref.get(cell_id, 0.0))
            c = float(cur.get(cell_id, 0.0))
            rows.append(
                {
                    "cell": cell_id,
                    "reference_share": r / max(ref.sum(), 1),
                    "current_share": c / max(cur.sum(), 1),
                    "share_delta": (c - r) / max(cur.sum() if c else 1, 1)
                    - (r / max(ref.sum(), 1)),
                }
            )
        rows.sort(key=lambda row: row["share_delta"], reverse=top)
        return rows[:10]
