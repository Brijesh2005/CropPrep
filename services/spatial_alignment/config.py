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
    ST_CACHE__ENABLED=true
    ST_ADMIN__ADMIN_DIR=D:/CropPrep/gis
    ST_TABULAR__TABLE=cropdata_updated.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from services.dataset_manager.config import _apply_case_insensitive, deep_merge, _parse_env
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


class AdminConfig(BaseModel):
    """Settings for administrative boundary support."""

    model_config = ConfigDict(extra="forbid")

    #: Boundary sources (shapefile/GeoJSON paths). Paths are resolved against
    #: ``admin_dir`` or the dataset root.
    boundaries: list[str] = Field(default_factory=list)
    #: Directory containing boundary files (mirrors Dataset Manager admin_dir).
    admin_dir: Path | None = None
    #: Column holding the feature name (village/taluk/district).
    name_column: str = "name"
    #: Column holding the administrative level, when present.
    level_column: str = "level"


class TabularConfig(BaseModel):
    """Settings for matching tabular agricultural records."""

    model_config = ConfigDict(extra="forbid")

    #: CSV file name to use as the agricultural record table. When None the
    #: table is auto-discovered through the Dataset Manager.
    table: str | None = None
    village_column: str = "village"
    district_column: str = "district"
    season_column: str = "season"
    year_column: str = "year"
    crop_column: str = "crop"
    yield_column: str = "yield"
    #: Explicit feature columns to expose; empty => all non-key columns.
    feature_columns: list[str] = Field(default_factory=list)
    #: Fall back to a district-level aggregate when no village row matches.
    fallback_to_district: bool = True


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

    merged = deep_merge(data, _parse_env(env_map, prefix=ENV_PREFIX))
    merged = _apply_case_insensitive(merged, StamConfig)
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
        "quality": {"max_temporal_gap_days": 60, "min_observations": 1,
                    "fail_below": 40.0},
        "cache": {"enabled": True, "observation_ttl_seconds": 3600,
                  "index_ttl_seconds": 86400},
        "admin": {"boundaries": [], "admin_dir": None, "name_column": "name",
                  "level_column": "level"},
        "tabular": {"table": None, "village_column": "village",
                    "district_column": "district", "season_column": "season",
                    "year_column": "year", "crop_column": "crop",
                    "yield_column": "yield", "feature_columns": [],
                    "fallback_to_district": True},
        "seasons": DEFAULT_SEASONS,
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
