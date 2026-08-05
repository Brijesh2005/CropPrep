"""GIS value objects (architecture contract)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from shared.enums import Season


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """A bare WGS84 coordinate the platform can resolve."""

    lon: float
    lat: float

    def __post_init__(self) -> None:
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon out of range [-180, 180]: {self.lon}")
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat out of range [-90, 90]: {self.lat}")


@dataclass(frozen=True, slots=True)
class ResolvedPlace:
    """Result of the reverse-geocoding step."""

    #: Display name (village, locality or district).
    name: str
    lon: float
    lat: float
    #: Source of the resolution (location_index.parquet / boundary hit / n/a).
    source: str = "unknown"
    #: 0..1 confidence of the match.
    confidence: float = 0.0
    #: Raw administrative hints from the index row.
    hints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdminContext:
    """Administrative hierarchy resolved from the boundary data."""

    district: str | None = None
    taluk: str | None = None
    village: str | None = None
    #: Geometry reference (boundary id / geometry id in the shapefile).
    geometry_id: str | None = None
    #: Raw properties captured at resolve time (for audit / display).
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "district": self.district,
            "taluk": self.taluk,
            "village": self.village,
            "geometry_id": self.geometry_id,
            "properties": self.properties,
        }


@dataclass(frozen=True, slots=True)
class HistoricalContext:
    """Historical context resolved for a point + date."""

    season: Season = Season.UNKNOWN
    #: Multi-year climatology summary (from historical_context.parquet).
    climatology: dict[str, Any] = field(default_factory=dict)
    #: Year the prediction targets (resolved from request date).
    year: int | None = None
    #: Optional planting-window hint used by the engine.
    planting_window: str | None = None


@dataclass(frozen=True, slots=True)
class GeoContext:
    """Final GIS resolution for a point + date, consumed by the services layer."""

    point: GeoPoint
    place: ResolvedPlace | None = None
    admin: AdminContext = field(default_factory=AdminContext)
    historical: HistoricalContext = field(default_factory=HistoricalContext)
    request_date: date = field(default_factory=date.today)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": {"lon": self.point.lon, "lat": self.point.lat},
            "place": (
                {
                    "name": self.place.name,
                    "source": self.place.source,
                    "confidence": self.place.confidence,
                    "hints": self.place.hints,
                }
                if self.place
                else None
            ),
            "admin": self.admin.to_dict(),
            "historical": {
                "season": self.historical.season.value,
                "year": self.historical.year,
                "climatology": self.historical.climatology,
                "planting_window": self.historical.planting_window,
            },
            "request_date": self.request_date.isoformat(),
            "extra": self.extra,
        }


__all__ = [
    "AdminContext",
    "GeoContext",
    "GeoPoint",
    "HistoricalContext",
    "ResolvedPlace",
]
