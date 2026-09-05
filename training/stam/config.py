"""STAM configuration (patch sizes, seasons, thresholds, caching).

Settings resolve with the same precedence as the Dataset Manager: environment
variables (prefix ``ST_``, nesting separated by ``__``) override a YAML file
(``ST_CONFIG_FILE`` / ``--config``) which overrides built-in defaults. Every
field is validated by pydantic.

Key options::

    ST_PATCH__SIZE=128
    ST_SPATIAL__MAX_SEARCH_RADIUS_KM=5.0
    ST_TEMPORAL__DEFAULT_SEASON=Kharif
    ST_TEMPORAL__TOLERANCE_DAYS=15
    ST_IMAGE__RESOLUTION=R10m
    ST_IMAGERY__MODE=window_days
    ST_IMAGERY__WINDOW_DAYS=180
    ST_IMAGERY__STRATEGY=closest_to_survey
    ST_CACHE__ENABLED=true
    ST_ADMIN__ADMIN_DIR=D:/CropPrep/gis
    ST_TABULAR__TABLE=cropdata_updated.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.config import apply_case_insensitive, deep_merge, parse_env
from .exceptions import StamConfigurationError

ENV_PREFIX = "ST_"

#: Default season calendar (Indian cropping seasons). Rabi crosses the
#: calendar-year boundary (Nov -> Mar), which the calendar handles explicitly.
DEFAULT_SEASONS = [
    {"name": "Kharif", "start_month": 6, "end_month": 10, "start_day": 1, "end_day": 31},
    {"name": "Rabi", "start_month": 11, "end_month": 3, "start_day": 1, "end_day": 31},
    {"name": "Summer", "start_month": 4, "end_month": 5, "start_day": 1, "end_day": 31},
]


class PatchConfig(BaseModel):
    """Settings for the spatial patch generator."""

    model_config = ConfigDict(extra="forbid")

    #: Edge length of the square patch in pixels (e.g. 128/224/256).
    size: int = Field(default=128, ge=8, le=2048)
    #: Padding mode for edge correction: ``"constant"`` or ``"reflect"``.
    pad_mode: str = Field(default="constant", pattern="^(constant|reflect)$")
    #: Fill value used with ``pad_mode="constant"``.
    pad_value: float = 0.0
    #: Shrink the patch at raster edges instead of failing when True.
    edge_correction: bool = True


class SpatialConfig(BaseModel):
    """Settings for spatial matching."""

    model_config = ConfigDict(extra="forbid")

    #: Maximum search radius for the nearest dataset location (km).
    max_search_radius_km: float = Field(default=5.0, ge=0.0)
    #: Hard threshold above which a location match is flagged low-confidence.
    distance_threshold_km: float = Field(default=5.0, ge=0.0)
    #: Leaf size for the KDTree (performance tuning).
    kdtree_leaf_size: int = Field(default=40, ge=2)
    #: Two dataset locations closer than this (metres) are considered the same
    #: point when the location catalog is built.
    duplicate_tolerance_m: float = Field(default=50.0, ge=0.0)


class TemporalConfig(BaseModel):
    """Settings for temporal matching."""

    model_config = ConfigDict(extra="forbid")

    #: Year used when the caller does not supply one (None => latest available).
    default_year: int | None = Field(default=None, ge=1950, le=2100)
    #: Season used when the caller does not supply one (None => infer from date).
    default_season: str | None = None
    #: Acceptable offset between a requested and an observed date (days).
    tolerance_days: int = Field(default=15, ge=0)
    #: Maximum allowed gap between consecutive observations (days).
    max_gap_days: int = Field(default=60, ge=0)
    #: YAML season-calendar file used by the :class:`SeasonResolver`
    #: (defaults to the bundled ``seasons.yaml``; ``ST_SEASONS_FILE`` wins).
    season_file: Path | None = None


class ImageConfig(BaseModel):
    """Settings for image matching / pairing."""

    model_config = ConfigDict(extra="forbid")

    #: Preferred resolution band (R10m / R20m).
    resolution: str = Field(default="R10m", pattern="^(R10m|R20m|UNKNOWN)$")
    #: Vegetation indices to align (NDVI, EVI).
    index_types: list[str] = Field(default_factory=lambda: ["NDVI", "EVI"])
    #: Require every observation date to have both NDVI and EVI.
    require_pairs: bool = True


class ImageryWindowConfig(BaseModel):
    """How the NDVI/EVI acquisition window is resolved around each sample.

    The legacy behaviour matches records inside the cropping-season calendar
    window (``mode="season"``). Seasonal-composite datasets such as the Kaggle
    NDVI/EVI mount hold only a handful of composite dates per year clustered in
    late-Apr/May and late-Oct, so a season window resolves exactly one composite
    for Kharif surveys. The window modes below widen acquisition while keeping
    every frame a REAL record that exists on disk:

    * ``"season"``        — legacy: the season calendar window for ``year``.
    * ``"window_days"``   — ``[survey_date - days, survey_date + days]``.
    * ``"crop_year"``     — ``[start_month .. start_month+span_months)`` of the
      survey reference's calendar year (season-agnostic crop-year context).

    ``max_observations`` trims the ordered real sequence to at most that many
    frames using ``strategy``; the sequence order itself is never altered and no
    date is fabricated, duplicated or zero-filled here.
    """

    model_config = ConfigDict(extra="forbid")

    #: Window resolution mode: ``season`` | ``window_days`` | ``crop_year``.
    mode: str = Field(default="season", pattern="^(season|window_days|crop_year)$")
    #: Half-width of the ``window_days`` window around the reference date.
    window_days: int = Field(default=180, ge=1)
    #: Month (1-12) anchoring a ``crop_year`` window start.
    start_month: int = Field(default=5, ge=1, le=12)
    #: Length of a ``crop_year`` window in months.
    span_months: int = Field(default=12, ge=1, le=36)
    #: Maximum number of real temporal frames kept per observation.
    max_observations: int = Field(default=8, ge=1, le=32)
    #: Frame-selection strategy when more dates exist than ``max_observations``.
    strategy: str = Field(
        default="closest_to_survey",
        pattern=(
            "^(closest_to_survey|evenly_spaced|quality_ranked|"
            "temporal_quality_combined|phenology_coverage)$"
        ),
    )
    #: Require every retained frame to carry both NDVI and EVI records.
    require_pairs: bool = True


class QualityConfig(BaseModel):
    """Settings for the quality-control pass."""

    model_config = ConfigDict(extra="forbid")

    max_temporal_gap_days: int = Field(default=60, ge=0)
    min_observations: int = Field(default=1, ge=0)
    #: Score below which an observation is treated as failed.
    fail_below: float = Field(default=40.0, ge=0.0, le=100.0)


class CacheConfig(BaseModel):
    """Settings for STAM caching."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    observation_ttl_seconds: int = Field(default=3600, ge=0)
    index_ttl_seconds: int = Field(default=86400, ge=0)


class BoundarySource(BaseModel):
    """One configured administrative boundary layer.

    ``path`` is a shapefile/GeoJSON path resolved against ``admin_dir`` or the
    dataset root. ``name_column``/``level`` override the global
    :class:`AdminConfig` defaults so every shapefile can use its own attribute
    (e.g. ``KGISDist_1`` for districts, ``KGISTalukN`` for taluks).
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    #: Attribute holding the feature name (falls back to AdminConfig.name_column).
    name_column: str | None = None
    #: Administrative level stamped onto every feature (falls back to the
    #: layer's own ``level_column`` when not given).
    level: str | None = None


class AdminConfig(BaseModel):
    """Settings for administrative boundary support."""

    model_config = ConfigDict(extra="forbid")

    #: Boundary sources (shapefile/GeoJSON paths with per-layer attributes).
    #: Plain string paths are accepted for backward compatibility.
    boundaries: list[BoundarySource] = Field(default_factory=list)
    #: Directory containing boundary files (mirrors Dataset Manager admin_dir).
    admin_dir: Path | None = None
    #: Default column holding the feature name (village/taluk/district).
    name_column: str = "name"
    #: Default column holding the administrative level, when present.
    level_column: str = "level"

    @field_validator("boundaries", mode="before")
    @classmethod
    def _coerce_boundaries(cls, value: Any) -> Any:
        """Coerce plain string paths into :class:`BoundarySource` entries."""
        if value is None:
            return []
        coerced: list[Any] = []
        for item in value:
            if isinstance(item, str):
                coerced.append({"path": item})
            elif isinstance(item, Mapping):
                coerced.append(dict(item))
            else:
                coerced.append(item)
        return coerced


class TabularTableConfig(BaseModel):
    """One tabular record table in the ordered multi-source fallback chain.

    Every (location, year, season) is matched against the first table before
    falling back to the next, so a fine-grained table (e.g.
    ``data_season.csv`` with village-level rows) wins over a coarse one
    (e.g. ICRISAT district-level data) whenever both could answer.
    """

    model_config = ConfigDict(extra="forbid")

    #: CSV file name inside the tabular datasets directory.
    name: str
    #: Column holding the village/locality name (matching level "village").
    village_column: str | None = None
    #: Column holding the taluk name (defaults to ``village_column``).
    taluk_column: str | None = None
    #: Column holding the district name (matching level "district").
    district_column: str | None = None
    #: Column holding the season (optional — annual tables skip it).
    season_column: str | None = None
    #: Column holding the year.
    year_column: str | None = None
    #: Column holding the crop label (optional — wide tables derive it).
    crop_column: str | None = None
    #: Column holding the yield label (optional — wide tables derive it).
    yield_column: str | None = None
    #: Explicit feature columns to expose; empty => all non-key columns.
    feature_columns: list[str] = Field(default_factory=list)
    #: Fall back to a district-level aggregate when no village row matches.
    fallback_to_district: bool = True
    #: Optional state filter (e.g. ICRISAT holds 20 states). When both are
    #: set only rows whose ``state_column`` equals ``state_value`` are matched.
    state_column: str | None = None
    state_value: str | None = None


class TabularConfig(BaseModel):
    """Settings for matching tabular agricultural records."""

    model_config = ConfigDict(extra="forbid")

    #: CSV file name to use as the agricultural record table. When None the
    #: table is auto-discovered through the Dataset Manager. Legacy single
    #: table option — prefer :attr:`tables` for multi-source fallback.
    table: str | None = None
    #: Ordered list of record tables. Each (location, year, season) is matched
    #: against the first table (village -> taluk -> district) before falling
    #: back to the next table.
    tables: list[TabularTableConfig] = Field(default_factory=list)
    village_column: str = "village"
    taluk_column: str | None = None
    district_column: str = "district"
    season_column: str = "season"
    year_column: str = "year"
    crop_column: str = "crop"
    yield_column: str = "yield"
    #: Explicit feature columns to expose; empty => all non-key columns.
    feature_columns: list[str] = Field(default_factory=list)
    #: Fall back to a district-level aggregate when no village row matches.
    fallback_to_district: bool = True

    def effective_tables(self) -> list[TabularTableConfig]:
        """The ordered tables to search, synthesised from legacy fields.

        When ``tables`` is configured it is returned as-is. Otherwise a single
        table is synthesised from the legacy top-level fields so existing
        ``table``-only configurations keep working unchanged.
        """
        if self.tables:
            return list(self.tables)
        if self.table:
            return [
                TabularTableConfig(
                    name=self.table,
                    village_column=self.village_column,
                    taluk_column=self.taluk_column,
                    district_column=self.district_column,
                    season_column=self.season_column,
                    year_column=self.year_column,
                    crop_column=self.crop_column,
                    yield_column=self.yield_column,
                    feature_columns=list(self.feature_columns),
                    fallback_to_district=self.fallback_to_district,
                )
            ]
        return []


class SeasonDef(BaseModel):
    """A configurable cropping season definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    start_month: int = Field(ge=1, le=12)
    end_month: int = Field(ge=1, le=12)
    start_day: int = Field(default=1, ge=1, le=31)
    end_day: int = Field(default=31, ge=1, le=31)

    @property
    def crosses_year_boundary(self) -> bool:
        """True when the season spans two calendar years (e.g. Rabi Nov->Mar)."""
        return self.start_month > self.end_month


class StamConfig(BaseModel):
    """Root STAM settings object."""

    model_config = ConfigDict(extra="forbid")

    patch: PatchConfig = Field(default_factory=PatchConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    imagery: ImageryWindowConfig = Field(default_factory=ImageryWindowConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    tabular: TabularConfig = Field(default_factory=TabularConfig)
    seasons: list[SeasonDef] = Field(default_factory=lambda: [SeasonDef(**s) for s in DEFAULT_SEASONS])

    def season_names(self) -> list[str]:
        return [s.name for s in self.seasons]


def load_stam_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> StamConfig:
    """Load and validate STAM settings (env > YAML > defaults).

    Args:
        config_path: Optional YAML file (falls back to ``ST_CONFIG_FILE``).
        env: Environment mapping (defaults to ``os.environ``).

    Raises:
        StamConfigurationError: For malformed YAML or invalid values.
    """
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("ST_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise StamConfigurationError(
                f"STAM configuration file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise StamConfigurationError(
                f"Malformed STAM YAML configuration: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise StamConfigurationError("STAM config root must be a mapping")
        data = raw

    merged = deep_merge(data, parse_env(env_map, prefix=ENV_PREFIX))
    merged = apply_case_insensitive(merged, StamConfig)
    try:
        return StamConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise StamConfigurationError(f"Invalid STAM configuration: {exc}") from exc


def save_stam_config_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default STAM configuration."""
    template = {
        "patch": {"size": 128, "pad_mode": "constant", "pad_value": 0.0,
                  "edge_correction": True},
        "spatial": {"max_search_radius_km": 5.0, "distance_threshold_km": 5.0,
                    "kdtree_leaf_size": 40, "duplicate_tolerance_m": 50.0},
        "temporal": {"default_year": None, "default_season": None,
                     "tolerance_days": 15, "max_gap_days": 60,
                     "season_file": None},
        "image": {"resolution": "R10m", "index_types": ["NDVI", "EVI"],
                  "require_pairs": True},
        "imagery": {"mode": "season", "window_days": 180, "start_month": 5,
                    "span_months": 12, "max_observations": 8,
                    "strategy": "closest_to_survey", "require_pairs": True},
        "quality": {"max_temporal_gap_days": 60, "min_observations": 1,
                    "fail_below": 40.0},
        "cache": {"enabled": True, "observation_ttl_seconds": 3600,
                  "index_ttl_seconds": 86400},
        "admin": {"boundaries": [], "admin_dir": None, "name_column": "name",
                  "level_column": "level"},
        "tabular": {"table": None, "tables": [],
                    "village_column": "village", "taluk_column": None,
                    "district_column": "district", "season_column": "season",
                    "year_column": "year", "crop_column": "crop",
                    "yield_column": "yield", "feature_columns": [],
                    "fallback_to_district": True},
        "seasons": DEFAULT_SEASONS,
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
