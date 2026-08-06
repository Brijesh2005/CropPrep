"""Temporal / sequence feature builder.

:class:`TemporalFeatureBuilder` summarises the *timing* of an observation:
how many observation dates fall inside the resolved season window, coverage of
the planting-to-harvest span, gaps between consecutive dates and the resolved
year/season. These features describe data availability over time, which the
image builder already partially covers — here they are expressed against the
season calendar rather than raw gap statistics.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .config import TemporalFeatureConfig
from .logger import get_logger

logger = get_logger("temporal")


class TemporalFeatureBuilder:
    """Build temporal-coverage features for an observation."""

    def __init__(self, config: TemporalFeatureConfig | None = None) -> None:
        self.config = config or TemporalFeatureConfig()

    def build(self, observation: Any, *, prefix: str = "tmp") -> dict[str, Any]:
        """Build temporal features for an observation.

        Args:
            observation: An :class:`AgriculturalObservation`.
            prefix: Modality prefix (``""`` disables prefixing).
        """
        features: dict[str, Any] = {}
        p = _pfx(prefix)
        temporal = observation.temporal
        dates = temporal.observation_dates or []

        features[p("year")] = _num(temporal.year)
        features[p("season")] = temporal.season
        features[p("date_count")] = _num(len(dates))
        features[p("within_season_count")] = _num(
            _within_season_count(dates, temporal.planting_start, temporal.harvest_end)
        )
        if temporal.planting_start is not None and temporal.harvest_end is not None:
            features[p("season_span_days")] = _num(
                (temporal.harvest_end - temporal.planting_start).days
            )
        else:
            features[p("season_span_days")] = None

        if dates:
            features[p("coverage_ratio")] = _num(
                len(dates) / max(1, _days_between(temporal.planting_start, temporal.harvest_end))
            )
            features[p("mean_gap_days")] = _num(_mean_gap(dates))
            features[p("max_gap_days")] = _num(_max_gap(dates))
            features[p("tolerance_days")] = _num(temporal.tolerance_days)
        else:
            features[p("coverage_ratio")] = None
            features[p("mean_gap_days")] = None
            features[p("max_gap_days")] = None
            features[p("tolerance_days")] = _num(temporal.tolerance_days)

        if self.config.include_dates:
            features[p("observation_dates")] = ";".join(d.isoformat() for d in dates)
        return features


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pfx(prefix: str) -> Any:
    def wrap(key: str) -> str:
        return f"{prefix}.{key}" if prefix else key

    return wrap


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _within_season_count(
    dates: list[date], planting_start: date | None, harvest_end: date | None
) -> int:
    if planting_start is None or harvest_end is None:
        return len(dates)
    return sum(1 for d in dates if planting_start <= d <= harvest_end)


def _days_between(start: date | None, end: date | None) -> int:
    if start is None or end is None:
        return 0
    return max(1, (end - start).days)


def _mean_gap(dates: list[date]) -> float:
    gaps = _gaps(dates)
    return sum(gaps) / len(gaps) if gaps else 0.0


def _max_gap(dates: list[date]) -> float:
    gaps = _gaps(dates)
    return max(gaps) if gaps else 0.0


def _gaps(dates: list[date]) -> list[float]:
    ordered = sorted(dates)
    return [float((b - a).days) for a, b in zip(ordered, ordered[1:])]
