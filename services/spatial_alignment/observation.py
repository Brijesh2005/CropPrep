"""Strongly-typed observation models for STAM.

The central output is :class:`AgriculturalObservation` — one unified
multimodal agricultural observation sample assembled from tabular features
plus an ordered NDVI/EVI image sequence for a location, year and season.

All models are pydantic v2 models: they validate on construction, serialize
to JSON/dict, and are safe to pass into training pipelines (Phase 4) without
any further transformation.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class GeographicPoint(BaseModel):
    """A point on Earth (WGS-84)."""

    lon: float = Field(ge=-180.0, le=180.0)
    lat: float = Field(ge=-90.0, le=90.0)


class AdminLocation(BaseModel):
    """Administrative hierarchy resolved for a point."""

    village: str | None = None
    taluk: str | None = None
    district: str | None = None
    state: str | None = None
    country: str | None = None
    #: The finest admin level actually matched (village/taluk/district).
    level: str | None = None
    #: Boundary source this resolution came from.
    source: str | None = None


class LocationInfo(BaseModel):
    """Resolved location for an observation."""

    lon: float
    lat: float
    #: Distance from the queried point to the nearest dataset location (km).
    distance_km: float | None = None
    dataset_location_id: str | None = None
    dataset_location_name: str | None = None
    admin: AdminLocation | None = None


class TemporalInfo(BaseModel):
    """Resolved temporal context for an observation."""

    year: int
    season: str | None = None
    season_months: tuple[int, int] | None = None
    observation_dates: list[date] = Field(default_factory=list)
    planting_start: date | None = None
    harvest_end: date | None = None
    tolerance_days: int = 0


class ImageRecordRef(BaseModel):
    """Reference to one image record (via the Dataset Manager metadata)."""

    path: str
    relative_path: str
    index_type: str = Field(pattern="^(NDVI|EVI|NONE)$")
    resolution: str = Field(pattern="^(R10m|R20m|UNKNOWN)$")
    observation_date: date | None = None
    year: int | None = None
    crs: str | None = None
    pixel_size: tuple[float, float] | None = None
    bounds: tuple[float, float, float, float] | None = None  # left, bottom, right, top
    #: Raster dimensions (used to reconstruct the affine transform for patches).
    width: int | None = None
    height: int | None = None


class ImagePairRef(BaseModel):
    """An NDVI/EVI pair observed on a single date."""

    date: date
    ndvi: ImageRecordRef | None = None
    evi: ImageRecordRef | None = None
    resolution: str = "UNKNOWN"
    crs: str | None = None
    #: Per-pair quality flags, e.g. ``{"paired": True, "missing": []}``.
    quality: dict[str, Any] = Field(default_factory=dict)


class SequenceInfo(BaseModel):
    """The ordered NDVI/EVI time series for the observation."""

    pairs: list[ImagePairRef] = Field(default_factory=list)
    sorted_dates: list[date] = Field(default_factory=list)
    resolution: str = "UNKNOWN"
    crs: str | None = None
    ndvi_paths: list[str] = Field(default_factory=list)
    evi_paths: list[str] = Field(default_factory=list)
    #: Day gaps between consecutive observation dates.
    gap_days: list[float] = Field(default_factory=list)


class HistoricalContext(BaseModel):
    """Multi-year temporal context built before inference for a location.

    Captures, for the resolved season, which historical years have satellite
    coverage plus the dataset / season-calendar versions that produced the
    context — the "same location + same season across all years" evidence
    gathered before STAM assembles the current observation.
    """

    #: Resolved season name (e.g. ``"Kharif"``).
    season: str | None = None
    #: The observation's resolved planting year.
    resolved_year: int | None = None
    #: Years in the catalog covering the same season.
    years: list[int] = Field(default_factory=list)
    #: Per-year record counts (``{"2020": {"records": 6, ...}}``).
    per_year: dict[str, dict[str, int]] = Field(default_factory=dict)
    total_records: int = 0
    #: Dataset Manager version that produced the context.
    dataset_version: str | None = None
    #: Season calendar version used to resolve the season.
    season_calendar_version: str | None = None
    #: Where the context came from (``"dataset_manager"`` / ``"unavailable"``).
    source: str | None = None

    @property
    def version(self) -> str | None:
        """A stable version string for the historical context."""
        if self.dataset_version is None:
            return None
        return f"{self.dataset_version}|{self.season_calendar_version or 'default'}"


class TabularFeatures(BaseModel):
    """The matched tabular agricultural record (features + label source)."""
    crop: str | None = None
    yield_value: float | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None
    #: How the record matched: "village", "district" or "none".
    matched_level: str = "none"

    @field_validator("yield_value", mode="before")
    @classmethod
    def _coerce_yield(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class QualityIssue(BaseModel):
    """A single quality finding for an observation."""

    code: str
    severity: str = Field(pattern="^(info|warning|error|critical)$")
    message: str
    detail: Any = None


class QualityReport(BaseModel):
    """Aggregated quality control output."""

    passed: bool
    overall_score: float = Field(ge=0.0, le=100.0)
    issues: list[QualityIssue] = Field(default_factory=list)

    def by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts


class AgriculturalObservation(BaseModel):
    """One unified multimodal agricultural observation sample.

    This is what STAM produces for every (location, year, season) query and
    what Phase 4 (feature engineering) and the AI module will consume. No AI
    model ever touches the raw datasets — every sample passes through STAM.
    """

    observation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=datetime.now)

    location: LocationInfo
    temporal: TemporalInfo
    tabular: TabularFeatures
    sequence: SequenceInfo
    quality: QualityReport

    #: Training label (crop) sourced from the tabular record.
    crop: str | None = None
    #: Training label (yield, kg/ha) sourced from the tabular record.
    yield_value: float | None = None

    #: Patch edge size configured for this observation.
    patch_size: int = 0
    #: Provenance: dataset manager version, metadata record count, cache hit.
    provenance: dict[str, Any] = Field(default_factory=dict)

    #: Multi-year historical context built before inference (same location +
    #: same season across all years). Present on every observation.
    historical_context: HistoricalContext | None = None
    #: Dataset Manager version snapshot at observation time.
    dataset_version: str | None = None
    #: Season calendar version used to resolve the season.
    season_calendar_version: str | None = None

    # -- Convenience accessors ------------------------------------------------ #

    @property
    def has_paired_images(self) -> bool:
        return any(p.ndvi is not None and p.evi is not None for p in self.sequence.pairs)

    def num_observations(self) -> int:
        return len(self.sequence.pairs)

    def to_train_dict(self) -> dict[str, Any]:
        """Compact dict for training pipelines (paths, labels, features)."""
        return {
            "observation_id": str(self.observation_id),
            "lon": self.location.lon,
            "lat": self.location.lat,
            "year": self.temporal.year,
            "season": self.temporal.season,
            "crop": self.crop,
            "yield_value": self.yield_value,
            "tabular": self.tabular.fields,
            "ndvi_paths": self.sequence.ndvi_paths,
            "evi_paths": self.sequence.evi_paths,
            "observation_dates": [d.isoformat() for d in self.temporal.observation_dates],
            "resolution": self.sequence.resolution,
            "crs": self.sequence.crs,
            "quality_score": self.quality.overall_score,
            "patch_size": self.patch_size,
        }
