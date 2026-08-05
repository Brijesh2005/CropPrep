"""Serialisable drift result types shared by every analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import numpy as np

SEVERITIES = ("low", "moderate", "high")


def rank_severity(severity: str) -> int:
    """Rank a severity label for aggregation (low < moderate < high)."""
    return {"low": 0, "moderate": 1, "high": 2}.get(severity, 0)


def _to_serialisable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    return value


@dataclass
class DriftResult:
    """Base class carrying a verdict for one drift dimension."""

    dimension: str
    severity: str = "low"
    drifted: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"invalid severity: {self.severity}")

    def to_dict(self) -> dict[str, Any]:
        return _to_serialisable(asdict(self))


@dataclass
class FeatureDriftResult(DriftResult):
    """One feature's distribution drift."""

    feature: str = ""
    dtype: str = "numeric"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_serialisable(asdict(self))


@dataclass
class LabelDriftResult(DriftResult):
    """Label / outcome distribution drift."""

    task: str = "classification"
    categories: list[dict[str, Any]] = field(default_factory=list)
    novelty: list[Any] = field(default_factory=list)
    vanished: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_serialisable(asdict(self))


@dataclass
class PredictionDriftResult(DriftResult):
    """Model-output drift (class probabilities / scalar predictions)."""

    mode: str = "classification"
    confidence_shift: float = 0.0
    entropy_shift: float = 0.0
    top_class_shifts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_serialisable(asdict(self))


@dataclass
class SpatialDriftResult(DriftResult):
    """Geographic coverage / request-location drift."""

    cell_size_degrees: float = 1.0
    num_cells_reference: int = 0
    num_cells_current: int = 0
    novel_cell_share: float = 0.0
    mean_nearest_neighbour_km: float = 0.0
    hot_cells: list[dict[str, Any]] = field(default_factory=list)
    cold_cells: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_serialisable(asdict(self))


@dataclass
class TemporalDriftResult(DriftResult):
    """Drift evolution over time windows + trend detection."""

    windows: list[dict[str, Any]] = field(default_factory=list)
    trend_slope: float = 0.0
    trend_p_value: float = 1.0
    trend_direction: str = "stable"
    episode_count: int = 0
    latest_severity: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return _to_serialisable(asdict(self))


@dataclass
class DriftReport:
    """Aggregate report produced by :class:`quality.drift.monitor.DriftMonitor`."""

    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    reference_samples: int = 0
    current_samples: int = 0
    overall_severity: str = "low"
    drifted: bool = False
    features: list[FeatureDriftResult] = field(default_factory=list)
    labels: LabelDriftResult | None = None
    predictions: PredictionDriftResult | None = None
    spatial: SpatialDriftResult | None = None
    temporal: TemporalDriftResult | None = None

    def summary(self) -> dict[str, Any]:
        """Compact human/machine summary of the verdict."""
        return _to_serialisable(
            {
                "generated_at": self.generated_at,
                "reference_samples": self.reference_samples,
                "current_samples": self.current_samples,
                "overall_severity": self.overall_severity,
                "drifted": self.drifted,
                "dimensions": {
                    "features": {
                        "total": len(self.features),
                        "drifted": sum(1 for f in self.features if f.drifted),
                        "high": sum(1 for f in self.features if f.severity == "high"),
                    },
                    "label": self.labels.to_dict() if self.labels else None,
                    "predictions": self.predictions.to_dict() if self.predictions else None,
                    "spatial": self.spatial.to_dict() if self.spatial else None,
                    "temporal": self.temporal.to_dict() if self.temporal else None,
                },
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_serialisable(asdict(self))
