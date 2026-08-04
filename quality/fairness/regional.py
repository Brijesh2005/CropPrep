"""Regional fairness — geographic performance disparity.

Groups model outcomes by administrative district (or any region label) and
reports per-region metrics, the global parity verdict, and per-region
lat/lon centroids so results can be plotted on a map.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .evaluator import FairnessEvaluator
from .config import FairnessConfig


class RegionalFairnessEvaluator:
    """Fairness analysis where the sensitive attribute is geographic region."""

    def __init__(self, config: FairnessConfig | None = None) -> None:
        self.config = config or FairnessConfig()
        self._evaluator = FairnessEvaluator(self.config)

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        regions: Sequence[str],
        *,
        region_centroids: Mapping[str, tuple[float, float]] | None = None,
        y_proba: Sequence[float] | None = None,
        task: str = "classification",
    ) -> dict[str, Any]:
        """Evaluate fairness across regions.

        Returns a dict with ``result`` (the parity verdicts), ``regions``
        (per-region metrics + centroid), and ``status``.
        """
        result = self._evaluator.evaluate(
            y_true,
            y_pred,
            {"region": regions},
            y_proba=y_proba,
            task=task,
        )
        centroids = region_centroids or {}
        region_rows = []
        for group in result.groups:
            if "status" in group.metrics:
                continue
            region_rows.append(
                {
                    "region": group.group,
                    "support": group.support,
                    "metrics": group.metrics,
                    "centroid_lon": centroids.get(group.group, (None, None))[0],
                    "centroid_lat": centroids.get(group.group, (None, None))[1],
                }
            )
        return {
            "status": result.overall_status,
            "verdicts": [v.to_dict() for v in result.verdicts],
            "regions": region_rows,
            "attribute": "region",
        }
