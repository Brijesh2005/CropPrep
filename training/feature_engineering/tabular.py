"""Tabular + location feature builder.

:class:`TabularFeatureBuilder` flattens one
:class:`~training.stam.observation.AgriculturalObservation` into a row of
tabular/geographic features: query-point location, resolved administrative
hierarchy, the resolved year/season and the matched tabular record (crop,
yield and feature fields). Training labels (``crop`` / ``yield_value``) are
included by default so the same builder serves both feature engineering and
label export.
"""

from __future__ import annotations

from typing import Any

from .config import TabularFeatureConfig
from .logger import get_logger

logger = get_logger("tabular")


class TabularFeatureBuilder:
    """Build a flat tabular feature row from an observation."""

    def __init__(self, config: TabularFeatureConfig | None = None) -> None:
        self.config = config or TabularFeatureConfig()

    def build(self, observation: Any, *, prefix: str = "tab") -> dict[str, Any]:
        """Flatten the observation into a ``{column: value}`` row.

        Args:
            observation: An :class:`AgriculturalObservation`.
            prefix: Modality prefix used when the caller enables prefixes
                (pass ``""`` to disable prefixing).
        """
        features: dict[str, Any] = {}
        p = _pfx(prefix)

        location = observation.location
        if self.config.include_location:
            features[p("lon")] = _num(location.lon)
            features[p("lat")] = _num(location.lat)
            features[p("distance_km")] = _num(location.distance_km)
            features[p("location_id")] = location.dataset_location_id
            features[p("location_name")] = location.dataset_location_name
            admin = location.admin
            features[p("village")] = admin.village if admin else None
            features[p("district")] = admin.district if admin else None
            features[p("state")] = admin.state if admin else None
            features[p("admin_level")] = admin.level if admin else None

        temporal = observation.temporal
        features[p("year")] = _num(temporal.year)
        features[p("season")] = temporal.season

        tabular = observation.tabular
        if self.config.include_fields and tabular.fields:
            label_cols = {str(c).lower() for c in self.config.label_columns}
            for key, value in tabular.fields.items():
                if str(key).lower() in label_cols:
                    continue
                features[p(str(key))] = _json_safe(value)
        features[p("matched_level")] = tabular.matched_level

        if self.config.include_labels:
            features[p("crop")] = observation.crop
            features[p("yield_value")] = _num(observation.yield_value)

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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)
