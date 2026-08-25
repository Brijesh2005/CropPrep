"""Spatial + temporal + tabular matching against the Dataset Manager.

This module contains:

* **Adapters** — production implementations of the STAM ports that talk to
  the Dataset Manager (the only sanctioned data access path):

  * :class:`DatasetManagerImageSource` — image metadata records.
  * :class:`DatasetManagerImageReader` — lazy windowed raster reads.
  * :class:`DatasetManagerTabularSource` — tabular agricultural records.
  * :class:`DatasetManagerBoundaryProvider` — admin boundary GeoDataFrames.
  * :class:`DatasetManagerLocationCatalog` — dataset location points.

* :class:`SpatialTemporalMatcher` — orchestrates spatial matching (nearest
  dataset point), administrative resolution, temporal resolution (year /
  season window) and tabular matching into one request pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.dataset_manager import DatasetManager
from training.dataset_manager.models import MetadataRecord

from .config import AdminConfig, StamConfig, TabularConfig
from .coordinate_transform import WGS84, transform_point
from .exceptions import (
    InvalidCoordinatesError,
    LocationNotFoundError,
    NoTabularRecordError,
)
from .interfaces import (
    BoundaryProvider,
    ImageMetadataSource,
    ImageReader,
    LocationCatalog,
    TabularSource,
)
from .logger import get_logger
from .name_aliases import SPECIAL_CASES, normalize_name
from .observation import (
    AdminLocation,
    ImageRecordRef,
    LocationInfo,
    TabularFeatures,
    TemporalInfo,
)
from .spatial_index import (
    BoundaryHit,
    BoundaryIndex,
    KDTreeSpatialIndex,
    LocationPoint,
    NearestMatch,
)
from .temporal_index import Season, SeasonCalendar, TemporalIndex

logger = get_logger("matcher")

#: Repository root (this file lives at <root>/training/stam/matcher.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Adapters (Dataset Manager backed)
# --------------------------------------------------------------------------- #


def _record_to_ref(record: MetadataRecord) -> ImageRecordRef:
    return ImageRecordRef(
        path=str(record.path),
        relative_path=record.relative_path,
        index_type=record.index_type.value,
        resolution=record.resolution.value,
        observation_date=record.observation_date,
        year=record.year,
        crs=record.crs,
        pixel_size=record.pixel_size,
        bounds=record.bounds,
        width=record.width,
        height=record.height,
    )


class DatasetManagerImageSource(ImageMetadataSource):
    """Image metadata records straight from the Dataset Manager store."""

    def __init__(self, manager: DatasetManager) -> None:
        self.manager = manager

    def query_images(
        self,
        *,
        index_type: str | None = None,
        resolution: str | None = None,
        year: int | None = None,
    ) -> list[ImageRecordRef]:
        filters: dict[str, Any] = {}
        if index_type is not None:
            filters["index_type"] = index_type
        if resolution is not None:
            filters["resolution"] = resolution
        if year is not None:
            filters["year"] = year
        records = self.manager.query_metadata(category="geotiff", **filters)
        return [_record_to_ref(r) for r in records]

    def image_metadata(self, path: str) -> ImageRecordRef:
        record = self.manager.get_metadata(path)
        if record is not None:
            return _record_to_ref(record)
        # Fall back to a live header read (still through the Dataset Manager).
        meta = self.manager.image_metadata(path)
        return ImageRecordRef(
            path=path,
            relative_path=str(Path(path).name),
            index_type=meta.get("index_type", "NONE"),
            resolution=meta.get("resolution", "UNKNOWN"),
            observation_date=None,
            year=meta.get("year"),
            crs=meta.get("crs"),
            pixel_size=tuple(meta["pixel_size"]) if meta.get("pixel_size") else None,
            bounds=tuple(meta["bounds"]) if meta.get("bounds") else None,
            width=meta.get("width"),
            height=meta.get("height"),
        )


class DatasetManagerImageReader(ImageReader):
    """Lazy windowed reads through the Dataset Manager."""

    def __init__(self, manager: DatasetManager) -> None:
        self.manager = manager

    def read_window(
        self, path: str, window: tuple[int, int, int, int], band: int = 1
    ) -> np.ndarray:
        return self.manager.load_image(path, window=window, band=band)

    def read_metadata(self, path: str) -> dict[str, Any]:
        return self.manager.image_metadata(path)


class DatasetManagerTabularSource(TabularSource):
    """Tabular agricultural records via the Dataset Manager.

    Records are searched across an ordered chain of tables (see
    :meth:`TabularConfig.effective_tables`): each (location, year, season) is
    matched against the first table at village -> taluk -> district level
    before falling back to the next table. Rows are matched by normalised name
    (case-insensitive, aliases + slash-separated alternates). The
    ``matched_level`` and ``__source_path``/``__source_table`` of the winning
    record are stamped so downstream quality checks can see exactly which
    source answered (see ST-Q-TAB-001).
    """

    def __init__(self, manager: DatasetManager, config: TabularConfig | None = None) -> None:
        self.manager = manager
        self.config = config or TabularConfig()
        self._frames: dict[str, tuple[TabularTableConfig, Path, pd.DataFrame]] = {}

    # -- Table discovery ------------------------------------------------------ #

    def _load_tables(self) -> list[tuple[TabularTableConfig, Path, pd.DataFrame]]:
        if self._frames:
            return list(self._frames.values())
        csvs = self.manager.list_csvs()
        if not csvs:
            raise NoTabularRecordError("No CSV files discovered via the Dataset Manager")
        tables = self.config.effective_tables()
        frames: list[tuple[TabularTableConfig, Path, pd.DataFrame]] = []
        if tables:
            for table_cfg in tables:
                path = self._resolve_table_path(csvs, table_cfg.name)
                frames.append((table_cfg, path, self.manager.load_csv(path)))
        else:
            # Backward compatibility: auto-discover the widest "master" CSV and
            # treat it as a single table using the legacy top-level columns.
            path = self._pick_widest(csvs)
            legacy = TabularTableConfig(
                name=path.name,
                village_column=self.config.village_column,
                taluk_column=self.config.taluk_column,
                district_column=self.config.district_column,
                season_column=self.config.season_column,
                year_column=self.config.year_column,
                crop_column=self.config.crop_column,
                yield_column=self.config.yield_column,
                feature_columns=list(self.config.feature_columns),
                fallback_to_district=self.config.fallback_to_district,
            )
            frames.append((legacy, path, self.manager.load_csv(path)))
        self._frames = {cfg.name: (cfg, path, frame) for cfg, path, frame in frames}
        return frames

    def _resolve_table_path(self, csvs: list[Path], name: str) -> Path:
        target = Path(name).name.lower()
        for path in csvs:
            if path.name.lower() == target:
                return path
        raise NoTabularRecordError(
            f"Configured tabular table not found: {name}",
            detail=[str(p) for p in csvs],
        )

    def _pick_widest(self, csvs: list[Path]) -> Path:
        """Auto-discovery: prefer the CSV exposing the most columns (the
        widest "master" table in a multi-file dataset)."""
        best, best_cols = None, -1
        for path in csvs:
            try:
                preview = self.manager.preview_csv(path, n_rows=1)
                cols = len(preview.columns)
            except Exception:  # noqa: BLE001
                cols = 0
            if cols > best_cols:
                best, best_cols = path, cols
        if best is None:
            raise NoTabularRecordError("No readable CSV table available")
        return best

    # -- Matching ------------------------------------------------------------- #

    def load_record(
        self,
        *,
        village: str | None,
        taluk: str | None,
        district: str | None,
        year: int,
        season: str | None,
    ) -> dict[str, Any] | None:
        """Best record across the table chain for the location/season/year.

        Tables are tried in configured order. Within each table the name is
        matched at village -> taluk -> district level and the first hit wins;
        when no table answers exactly, the legacy behaviour of returning the
        first row of the year/season subset is preserved (matched_level
        ``"none"``, so ST-Q-TAB-001 still rejects it).
        """
        tables = self._load_tables()
        if not tables:
            return None

        fabrication: tuple[pd.DataFrame, TabularTableConfig, Path] | None = None
        for table_cfg, path, frame in tables:
            if frame is None or len(frame) == 0:
                continue
            subset = self._subset(frame, table_cfg, year, season)
            if subset is None or len(subset) == 0:
                continue
            matched = self._match_level(subset, table_cfg, village, taluk, district)
            if matched is not None:
                row, level = matched
                return self._to_record(row, level, table_cfg, path)
            if fabrication is None:
                # First table with any year/season rows provides the last-resort
                # fabricated record (flagged: it fails ST-Q-TAB-001).
                fabrication = (subset, table_cfg, path)
        if fabrication is not None:
            subset, table_cfg, path = fabrication
            return self._to_record(subset.iloc[0], "none", table_cfg, path)
        return None

    def available_years(self) -> list[int]:
        years: list[int] = []
        for table_cfg, path, frame in self._load_tables():
            if table_cfg.year_column and table_cfg.year_column in frame.columns:
                for value in frame[table_cfg.year_column]:
                    try:
                        years.append(int(float(value)))
                    except (TypeError, ValueError):
                        continue
        return sorted(set(years))

    # -- Internals ------------------------------------------------------------ #

    def _subset(
        self,
        frame: pd.DataFrame,
        table_cfg: TabularTableConfig,
        year: int,
        season: str | None,
    ) -> pd.DataFrame | None:
        """Rows of ``frame`` for the requested state/year/season."""
        subset = frame
        if table_cfg.state_column and table_cfg.state_value:
            if table_cfg.state_column in subset.columns:
                state = str(table_cfg.state_value).strip().lower()
                subset = subset[
                    subset[table_cfg.state_column].astype(str).str.strip().str.lower() == state
                ]
        if table_cfg.year_column and table_cfg.year_column in subset.columns:
            subset = _filter_equals(subset, table_cfg.year_column, year)
        if table_cfg.season_column and table_cfg.season_column in subset.columns and season:
            subset = _filter_contains(subset, table_cfg.season_column, season)
        return subset

    def _match_level(
        self,
        subset: pd.DataFrame,
        table_cfg: TabularTableConfig,
        village: str | None,
        taluk: str | None,
        district: str | None,
    ) -> tuple[pd.Series, str] | None:
        village_col = table_cfg.village_column
        taluk_col = table_cfg.taluk_column or table_cfg.village_column
        district_col = table_cfg.district_column
        if village and village_col and village_col in subset.columns:
            row = _first_contains(subset, village_col, village)
            if row is not None:
                return row, "village"
        # Taluk-level attempt (e.g. Madikeri) before the district fallback.
        if taluk and taluk_col and taluk_col in subset.columns and taluk != village:
            row = _first_contains(subset, taluk_col, taluk)
            if row is not None:
                return row, "taluk"
        if table_cfg.fallback_to_district and district and district_col and district_col in subset.columns:
            row = _first_contains(subset, district_col, district)
            if row is not None:
                return row, "district"
        return None

    def _to_record(
        self,
        row: pd.Series,
        level: str,
        table_cfg: TabularTableConfig,
        path: Path,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for column, value in row.items():
            record[str(column)] = _json_safe(value)
        record["__matched_level"] = level
        record["__source_path"] = str(path)
        record["__source_table"] = table_cfg.name
        record["__feature_columns"] = list(table_cfg.feature_columns)
        if table_cfg.crop_column and table_cfg.crop_column in row:
            record["__crop"] = _json_safe(row[table_cfg.crop_column])
        if table_cfg.yield_column and table_cfg.yield_column in row:
            record["__yield"] = _json_safe(row[table_cfg.yield_column])
        if not table_cfg.crop_column or not table_cfg.yield_column:
            # Wide-format table (e.g. ICRISAT "<CROP> AREA/YIELD" triples):
            # derive the dominant crop (largest planted area) and its yield.
            crop, area, yield_value = _dominant_crop_yield(row)
            record.setdefault("__crop", crop)
            record.setdefault("__yield", yield_value)
            record["__derived_fields"] = {
                "crop": crop,
                "yield": yield_value,
                "area": area,
                "district": (
                    _json_safe(row.get(table_cfg.district_column))
                    if table_cfg.district_column
                    else None
                ),
                "year": (
                    _json_safe(row.get(table_cfg.year_column))
                    if table_cfg.year_column
                    else None
                ),
            }
        return record


class DatasetManagerBoundaryProvider(BoundaryProvider):
    """Admin boundaries loaded through the Dataset Manager ``load_geometries``."""

    def __init__(self, manager: DatasetManager, config: AdminConfig | None = None) -> None:
        self.manager = manager
        self.config = config or AdminConfig()
        self._cache: Any | None = None

    def boundaries(self) -> Any:
        """Return a combined GeoDataFrame (EPSG:4326) of all configured sources.

        Every source is normalised to the canonical ``name``/``level`` columns
        using its own ``name_column``/``level`` metadata so a single schema
        feeds the boundary + location indexes (district and taluk layers use
        different KGIS attributes).
        """
        if self._cache is not None:
            return self._cache
        import geopandas as gpd

        sources = list(self.config.boundaries)
        if not sources:
            logger.info("No admin boundary sources configured")
            return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

        frames: list[Any] = []
        for source in sources:
            path = self._resolve_source(source.path)
            gdf = self.manager.load_geometries(path)
            if gdf.crs is not None:
                gdf = gdf.to_crs(4326)
            frames.append(_normalise_boundary_frame(gdf, source, self.config))
        combined = pd.concat(frames, ignore_index=True)
        if combined.crs is None:
            combined = combined.set_crs(4326)
        self._cache = combined
        return combined

    def _resolve_source(self, source: str) -> Path:
        candidate = Path(source)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        roots: list[Path] = []
        if self.config.admin_dir is not None:
            roots.append(Path(self.config.admin_dir))
        roots.append(self.manager.settings.dataset_root)
        # Repo-relative paths (e.g. ``application/gis/...``) work regardless
        # of the CWD — the Dataset Manager authorises them via admin_dir.
        roots.append(_REPO_ROOT)
        for root in roots:
            probe = root / source
            if probe.exists():
                return probe
        return candidate  # let load_geometries produce a precise error


class DatasetManagerLocationCatalog(LocationCatalog):
    """Dataset location points from admin boundaries + image record centroids."""

    def __init__(
        self,
        manager: DatasetManager,
        config: StamConfig,
        boundary_provider: BoundaryProvider,
    ) -> None:
        self.manager = manager
        self.config = config
        self.boundary_provider = boundary_provider

    def points(self) -> list[LocationPoint]:
        points: list[LocationPoint] = []

        # 1. Manual locations that cannot be resolved from boundary files
        #    (e.g. Kasaragodu, outside the Karnataka shapefiles).
        for raw_name, special in SPECIAL_CASES.items():
            if special.get("type") != "manual_point":
                continue
            lon = special.get("lon")
            lat = special.get("lat")
            if lon is None or lat is None:
                continue
            points.append(
                LocationPoint(
                    id=f"manual:{raw_name}",
                    name=raw_name,
                    lon=float(lon),
                    lat=float(lat),
                    meta={"source": "manual", "status": special.get("status", "manual")},
                )
            )

        # 2. Admin boundary centroids (villages / taluks / districts).
        gdf = self.boundary_provider.boundaries()
        if gdf is not None and len(gdf):
            name_col = self.config.admin.name_column
            level_col = self.config.admin.level_column
            for index, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                centroid = geom.representative_point() if not geom.is_valid else geom.centroid
                points.append(
                    LocationPoint(
                        id=f"boundary:{index}",
                        name=str(row.get(name_col, f"feature-{index}")),
                        lon=float(centroid.x),
                        lat=float(centroid.y),
                        meta={
                            "level": str(row.get(level_col)) if level_col in row else None,
                            "source": "boundary",
                            "attributes": {k: _json_safe(v) for k, v in row.items() if k != "geometry"},
                        },
                    )
                )

        # 3. Image record centroids (actual observation footprint).
        try:
            images = self.manager.query_metadata(category="geotiff")
        except Exception:  # noqa: BLE001 - image metadata may be unavailable
            images = []
        for record in images:
            if not record.bounds:
                continue
            left, bottom, right, top = record.bounds
            x, y = (left + right) / 2.0, (bottom + top) / 2.0
            # `record.bounds` is in the raster's native CRS (e.g. UTM metres
            # such as EPSG:32643), not WGS-84 degrees. STAM's public
            # boundary always deals in (lon, lat) WGS-84 — see
            # coordinate_transform.py — so the centroid must be reprojected
            # before being stored as a LocationPoint, otherwise every
            # downstream coordinate-range check (ST-COORD-001) fails.
            if record.crs:
                lon, lat = transform_point(record.crs, WGS84, x, y)
            else:
                lon, lat = x, y
            points.append(
                LocationPoint(
                    id=f"image:{record.relative_path}",
                    name=Path(record.relative_path).stem,
                    lon=float(lon),
                    lat=float(lat),
                    meta={"source": "image", "index_type": record.index_type.value,
                          "year": record.year},
                )
            )
        return points


# --------------------------------------------------------------------------- #
# Matcher
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TemporalContext:
    """Resolved temporal context for a query."""

    year: int
    season: Season | None
    reference_date: date | None

    @property
    def planting_start(self) -> date | None:
        return self.season.start if self.season else None

    @property
    def harvest_end(self) -> date | None:
        return self.season.end if self.season else None


class SpatialTemporalMatcher:
    """Coordinates spatial + temporal + tabular matching.

    Args:
        manager: The Dataset Manager (sole data access path).
        config: Validated STAM settings.
        image_source / image_reader / tabular_source / boundary_provider /
        location_catalog: Optional adapter overrides (injected for tests).
        spatial_index / boundary_index / calendar: Optional prebuilt indexes.
    """

    def __init__(
        self,
        manager: DatasetManager,
        config: StamConfig,
        *,
        image_source: ImageMetadataSource | None = None,
        image_reader: ImageReader | None = None,
        tabular_source: TabularSource | None = None,
        boundary_provider: BoundaryProvider | None = None,
        location_catalog: LocationCatalog | None = None,
        spatial_index: KDTreeSpatialIndex | None = None,
        boundary_index: BoundaryIndex | None = None,
        calendar: SeasonCalendar | None = None,
    ) -> None:
        self.manager = manager
        self.config = config
        self.image_source = image_source or DatasetManagerImageSource(manager)
        self.image_reader = image_reader or DatasetManagerImageReader(manager)
        self.boundary_provider = boundary_provider or DatasetManagerBoundaryProvider(
            manager, config.admin
        )
        self.tabular_source = tabular_source or DatasetManagerTabularSource(
            manager, config.tabular
        )
        self.location_catalog = location_catalog or DatasetManagerLocationCatalog(
            manager, config, self.boundary_provider
        )
        self.spatial_index = spatial_index or KDTreeSpatialIndex(
            max_radius_km=config.spatial.max_search_radius_km,
            duplicate_tolerance_m=config.spatial.duplicate_tolerance_m,
            leaf_size=config.spatial.kdtree_leaf_size,
        )
        self.boundary_index = boundary_index or BoundaryIndex(
            name_column=config.admin.name_column,
            level_column=config.admin.level_column,
        )
        self.calendar = calendar or SeasonCalendar(config.seasons)
        self._initialized = False

    # -- Initialization ------------------------------------------------------- #

    def initialize(self) -> "SpatialTemporalMatcher":
        """Load boundaries and build the spatial + boundary indexes."""
        gdf = self.boundary_provider.boundaries()
        self.boundary_index.build(gdf)
        points = self.location_catalog.points()
        self.spatial_index.build(points)
        self._initialized = True
        logger.info(
            "Matcher initialized",
            extra={
                "boundaries": len(self.boundary_index),
                "locations": len(self.spatial_index),
            },
        )
        return self

    def _require_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    # -- Spatial -------------------------------------------------------------- #

    def find_nearest(
        self, lon: float, lat: float, *, max_radius_km: float | None = None
    ) -> NearestMatch:
        """Nearest dataset location to ``(lon, lat)`` within the radius."""
        _validate_coordinates(lon, lat)
        self._require_initialized()
        return self.spatial_index.nearest_one(lon, lat)

    def resolve_admin(self, lon: float, lat: float) -> AdminLocation | None:
        """Administrative hierarchy containing the point, if boundaries exist."""
        self._require_initialized()
        if not self.boundary_index.is_built:
            return None
        hits = self.boundary_index.find_containing_all(lon, lat)
        if not hits:
            return None
        return _admin_from_hits(hits)

    def location_info(self, lon: float, lat: float) -> LocationInfo:
        """Assemble the full :class:`LocationInfo` for a query point."""
        _validate_coordinates(lon, lat)
        self._require_initialized()
        match = self.spatial_index.nearest_one(lon, lat)
        admin = self.resolve_admin(lon, lat)
        return LocationInfo(
            lon=lon,
            lat=lat,
            distance_km=round(match.distance_km, 4),
            dataset_location_id=match.point.id,
            dataset_location_name=match.point.name,
            admin=admin,
        )

    # -- Temporal ------------------------------------------------------------- #

    def resolve_temporal(
        self,
        *,
        year: int | None = None,
        season: str | None = None,
        reference_date: date | None = None,
    ) -> TemporalContext:
        """Resolve year + season (and the planting/harvest window).

        Rules:
        * Year defaults to the configured default, then the latest available.
        * Season defaults to the configured default, then is inferred from the
          reference date when provided.
        """
        self._require_initialized()
        season_name = season or self.config.temporal.default_season
        resolved_season: Season | None = None

        if season_name:
            if not self.calendar.has_season(season_name):
                raise ValueError(f"Unknown season: {season_name}")
            if year is not None:
                resolved_season = self.calendar.season_window(season_name, year)
            elif reference_date is not None:
                match = self.calendar.season_for_date(reference_date)
                if match is not None:
                    resolved_season = match[0]
        elif reference_date is not None:
            match = self.calendar.season_for_date(reference_date)
            if match is not None:
                resolved_season = match[0]
                season_name = match[0].name

        if year is None:
            year = self.config.temporal.default_year or self._latest_year(
                resolved_season
            )
        if resolved_season is None and season_name and year is not None:
            resolved_season = self.calendar.season_window(season_name, year)

        return TemporalContext(year=year, season=resolved_season, reference_date=reference_date)

    def _latest_year(self, season: Season | None) -> int:
        years = self.tabular_source.available_years()
        if years:
            return max(years)
        if season is not None:
            return max(season.year, 0) or 2025  # fallback
        return 2025

    # -- Tabular -------------------------------------------------------------- #

    def match_tabular(
        self,
        *,
        village: str | None,
        district: str | None,
        taluk: str | None = None,
        year: int,
        season: str | None,
    ) -> TabularFeatures | None:
        """Best tabular record for the location/season/year."""
        record = self.tabular_source.load_record(
            village=village,
            taluk=taluk,
            district=district,
            year=year,
            season=season,
        )
        if record is None:
            return None
        crop = record.get("__crop")
        yield_value = _json_safe(record.get("__yield"))
        try:
            yield_value = float(yield_value) if yield_value not in (None, "") else None
        except (TypeError, ValueError):
            yield_value = None

        fields = {k: v for k, v in record.items() if not k.startswith("__")}
        feature_cols = record.get("__feature_columns")
        if feature_cols:
            fields = {k: fields[k] for k in feature_cols if k in fields}
        elif record.get("__derived_fields"):
            # Wide-format tables keep only the derived (crop/area/yield) record.
            fields = dict(record["__derived_fields"])
        return TabularFeatures(
            crop=str(crop) if crop is not None else None,
            yield_value=yield_value,
            fields=fields,
            source_path=record.get("__source_path"),
            matched_level=record.get("__matched_level", "none"),
        )

    # -- Images --------------------------------------------------------------- #

    def match_images(
        self,
        *,
        year: int,
        season: Season | None,
        resolution: str | None = None,
    ) -> tuple[list[ImageRecordRef], list[ImageRecordRef]]:
        """NDVI + EVI records whose observation date falls in the season window.

        When no season window exists (custom seasons off), records are matched
        by year.
        """
        resolution = resolution or self.config.image.resolution
        ndvi = self.image_source.query_images(index_type="NDVI", resolution=resolution)
        evi = self.image_source.query_images(index_type="EVI", resolution=resolution)

        if season is not None:
            contains = self.calendar.contains
            ndvi = [
                r for r in ndvi
                if (r.observation_date and contains(season, r.observation_date))
                or (r.observation_date is None and r.year == year)
            ]
            evi = [
                r for r in evi
                if (r.observation_date and contains(season, r.observation_date))
                or (r.observation_date is None and r.year == year)
            ]
        else:
            ndvi = [r for r in ndvi if r.year == year]
            evi = [r for r in evi if r.year == year]
        return ndvi, evi

    # -- Indexes -------------------------------------------------------------- #

    def spatial_stats(self) -> dict[str, int]:
        return {"locations": len(self.spatial_index), "boundaries": len(self.boundary_index)}

    @property
    def initialized(self) -> bool:
        return self._initialized


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _validate_coordinates(lon: float, lat: float) -> None:
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise InvalidCoordinatesError(
            f"Out-of-range coordinates: lon={lon}, lat={lat}"
        )


def _admin_from_hits(hits: list[BoundaryHit]) -> AdminLocation:
    by_level: dict[str, BoundaryHit] = {}
    for hit in hits:
        level = (hit.level or "").lower()
        if level in {"village", "taluk", "district", "state", "country"}:
            by_level[level] = hit
    village = by_level.get("village")
    taluk = by_level.get("taluk")
    district = by_level.get("district")
    state = by_level.get("state")
    country = by_level.get("country")

    level_hit = village or taluk or district or state or country
    return AdminLocation(
        village=village.name if village else None,
        taluk=taluk.name if taluk else None,
        district=district.name if district else None,
        state=state.name if state else None,
        country=country.name if country else None,
        level=level_hit.level.lower() if level_hit else None,
        source="boundary",
    )


def _filter_equals(frame: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    mask = frame[column].astype(str).str.strip().str.lower() == str(value).strip().lower()
    return frame[mask]


def _filter_contains(frame: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    mask = frame[column].astype(str).str.contains(
        str(value).strip(), case=False, na=False
    )
    return frame[mask]


def _first_contains(frame: pd.DataFrame, column: str, value: Any) -> pd.Series | None:
    """First row whose ``column`` name matches ``value``.

    Both sides are normalised through :func:`normalize_name` (aliases,
    casing, parenthetical qualifiers) so a KGIS boundary name like
    ``"Dakshina Kannada"`` matches the CSV's ``"Mangalore"``. Exact match is
    preferred; slash-separated alternates (e.g. ICRISAT ``"Gulbarga /
    Kalaburagi"``) are tried next; a substring match on the normalised names
    is the final fallback.
    """
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None
    norm = normalize_name(raw)
    names = frame[column].astype(str)
    if norm:
        norm_names = names.map(normalize_name)
        exact = norm_names == norm
        if exact.any():
            return frame[exact].iloc[0]
        # Slash / pipe separated alternates, normalised per part.
        hits = norm_names.map(
            lambda cand: any(
                normalize_name(part.strip()) == norm
                for part in re.split(r"[/|]", cand)
                if part.strip()
            )
        )
        if hits.any():
            return frame[hits].iloc[0]
        contains = norm_names.str.contains(norm, regex=False, case=False, na=False)
        if contains.any():
            return frame[contains].iloc[0]
    mask = names.str.contains(raw, case=False, na=False)
    if mask.any():
        return frame[mask].iloc[0]
    return None


def _normalise_boundary_frame(
    gdf: Any, source: Any, config: AdminConfig
) -> Any:
    """Copy ``gdf`` into a canonical ``name``/``level`` schema.

    ``source.name_column`` (falling back to ``config.name_column``) is copied
    into a ``name`` column and ``source.level`` is stamped onto every feature
    so the combined boundary frame has one uniform schema.
    """
    frame = gdf.copy()
    name_col = source.name_column or config.name_column
    if name_col != "name":
        if name_col in frame.columns:
            frame["name"] = frame[name_col].astype(str)
        else:
            logger.warning(
                "Boundary source missing name column",
                extra={"path": source.path, "column": name_col},
            )
            frame["name"] = ""
    if source.level is not None:
        frame["level"] = str(source.level)
    elif "level" not in frame.columns:
        frame["level"] = None
    return frame


#: Wide-format ICRISAT-style column patterns ("RICE AREA (1000 ha)").
_AREA_RE = re.compile(r"^(.*?)\s+AREA\s*\(", re.IGNORECASE)
_YIELD_RE = re.compile(r"^(.*?)\s+YIELD\s*\(", re.IGNORECASE)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dominant_crop_yield(row: pd.Series) -> tuple[str | None, float | None, float | None]:
    """Derive (crop, area, yield) from a wide-format row.

    ICRISAT-style tables store one ``<CROP> AREA / PRODUCTION / YIELD`` column
    triple per crop. The dominant crop (largest planted area, ignoring the
    ``-1`` missing sentinel) is selected and its yield returned. When no
    usable AREA column exists the crop with the highest YIELD is used.
    """
    area_cols: dict[str, str] = {}
    yield_cols: dict[str, str] = {}
    for column in row.index:
        name = str(column)
        match = _AREA_RE.match(name)
        if match:
            area_cols[match.group(1).strip()] = name
            continue
        match = _YIELD_RE.match(name)
        if match:
            yield_cols[match.group(1).strip()] = name

    best: tuple[float, str] | None = None
    for crop, column in area_cols.items():
        value = _to_float(row.get(column))
        if value is None or value <= 0:
            continue
        if best is None or value > best[0]:
            best = (value, crop)
    if best is not None:
        area, crop = best
        yield_col = yield_cols.get(crop)
        yield_value = _to_float(row.get(yield_col)) if yield_col else None
        return crop.title(), area, yield_value

    best_yield: tuple[float, str] | None = None
    for crop, column in yield_cols.items():
        value = _to_float(row.get(column))
        if value is None or value <= 0:
            continue
        if best_yield is None or value > best_yield[0]:
            best_yield = (value, crop)
    if best_yield is not None:
        yield_value, crop = best_yield
        return crop.title(), None, yield_value
    return None, None, None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)
