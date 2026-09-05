"""The STAM facade — public API of the Spatial-Temporal Alignment Module.

STAM is the research contribution that transforms **tabular data + satellite
images** into one unified :class:`AgriculturalObservation` per (location, year,
season). Every training sample in the CropFusion pipeline must pass through
STAM; no AI model ever touches the raw datasets.

Public API::

    stam = STAM(manager, StamConfig(...))
    stam.initialize()
    obs = stam.build_observation(lon, lat)                    # farmer path
    obs = stam.build_observation(lon, lat, year=2020, season="Kharif")  # research
    nearest = stam.find_nearest(lon, lat)
    seq = stam.build_sequence(lon, lat, year=2020, season="Kharif")
    patch = stam.get_patch(image_path, lon, lat, size=128)
    report = stam.validate(obs)
    summary = stam.summary()

The farmer path needs only a location: the :class:`SeasonResolver` infers the
season from today's date (YAML-configurable calendar) and a multi-year
:class:`~training.stam.observation.HistoricalContext` (same
location + same season) is attached to every observation.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from training.dataset_manager import DatasetManager

from .cache import DatasetManagerStamCache
from .config import StamConfig, load_stam_config
from .coordinate_transform import WGS84, transform_point
from .exceptions import PairingError, StamError, NotInitializedError
from .historical_context import HistoricalContextBuilder
from .interfaces import StamCache
from .logger import get_logger
from .matcher import SpatialTemporalMatcher, TemporalContext
from .name_aliases import district_to_csv, normalize_name, resolve_location
from .observation import (
    AgriculturalObservation,
    LocationInfo,
    QualityReport,
    SequenceInfo,
    TemporalInfo,
)
from .patch_generator import RasterPatch, SpatialPatchGenerator
from .season_resolver import SeasonResolver
from .sequence_builder import ObservationSequenceBuilder, SequenceBuildResult
from .validators import assess_quality

logger = get_logger("stam")

__all__ = ["STAM"]


class STAM:
    """Spatial-Temporal Alignment Module facade.

    Args:
        manager: The Dataset Manager (sole data access path).
        config: Validated :class:`StamConfig`. Built from defaults when None.
        matcher: Optional pre-wired :class:`SpatialTemporalMatcher`.
        sequence_builder: Optional :class:`ObservationSequenceBuilder`.
        patch_generator: Optional :class:`SpatialPatchGenerator`.
        cache: Optional :class:`StamCache`.
        season_resolver: Optional :class:`SeasonResolver` (built from config).
        historical_context_builder: Optional :class:`HistoricalContextBuilder`.
    """

    def __init__(
        self,
        manager: DatasetManager,
        config: StamConfig | None = None,
        *,
        matcher: SpatialTemporalMatcher | None = None,
        sequence_builder: ObservationSequenceBuilder | None = None,
        patch_generator: SpatialPatchGenerator | None = None,
        cache: StamCache | None = None,
        season_resolver: SeasonResolver | None = None,
        historical_context_builder: HistoricalContextBuilder | None = None,
    ) -> None:
        self.manager = manager
        self.config = config or StamConfig()
        self.cache = cache or DatasetManagerStamCache(
            manager,
            enabled=self.config.cache.enabled,
            default_ttl_seconds=self.config.cache.observation_ttl_seconds,
        )
        self.season_resolver = season_resolver or SeasonResolver.from_config(
            self.config
        )
        self.matcher = matcher or SpatialTemporalMatcher(
            manager, self.config, calendar=self.season_resolver.calendar
        )
        self.sequence_builder = sequence_builder or ObservationSequenceBuilder(
            require_pairs=self.config.image.require_pairs,
            max_gap_days=self.config.quality.max_temporal_gap_days,
        )
        self.patch_generator = patch_generator or SpatialPatchGenerator(
            self.matcher.image_reader,
            self.matcher.image_source,
            default_size=self.config.patch.size,
            default_pad_mode=self.config.patch.pad_mode,
            default_pad_value=self.config.patch.pad_value,
            edge_correction=self.config.patch.edge_correction,
        )
        self.historical_context_builder = (
            historical_context_builder
            or HistoricalContextBuilder(manager, self.config, self.season_resolver)
        )
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        manager: DatasetManager,
        config_path: str | None = None,
    ) -> "STAM":
        """Build a STAM from a YAML config file / ``ST_*`` environment vars."""
        return cls(manager, load_stam_config(config_path))

    def initialize(self) -> "STAM":
        """Load boundaries and build the spatial + temporal indexes."""
        self.matcher.initialize()
        self._initialized = True
        if self.config.cache.enabled:
            self.cache.set(
                self.cache.index_key("matcher"),
                {"initialized": True},
                ttl_seconds=self.config.cache.index_ttl_seconds,
            )
        logger.info("STAM initialized")
        return self

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise NotInitializedError(
                "STAM.initialize() must be called before build_observation()"
            )

    def build_observation(
        self,
        lon: float,
        lat: float,
        *,
        year: int | None = None,
        season: str | None = None,
        month: int | None = None,
        reference_date: date | None = None,
        resolution: str | None = None,
        use_cache: bool = True,
    ) -> AgriculturalObservation:
        """Assemble one :class:`AgriculturalObservation` for a point.

        Args:
            lon / lat: Query point (WGS-84) — from GPS or a map click.
            year: Target year (default: configured / latest available). Omit
                for the farmer path where the season is auto-resolved.
            season: Season name (default: configured / inferred from date).
            month: Optional month used to infer the season when not given.
            reference_date: Explicit reference date (overrides ``month``).
            resolution: Image resolution band (default R10m).
            use_cache: Serve from / store in the observation cache.

        Returns:
            A fully typed :class:`AgriculturalObservation`. When neither
            ``year`` nor ``season`` is supplied the season is inferred from
            the calendar date (SeasonResolver) and the year defaults to the
            latest year with data — the farmer workflow needs only a location.
        """
        self._require_initialized()
        if reference_date is None and month is not None:
            base_year = year or self.config.temporal.default_year or 2020
            reference_date = date(base_year, month, 15)

        year = year or self.config.temporal.default_year
        season_name = season or self.config.temporal.default_season
        if season_name is None:
            # Farmer path: infer the season from the calendar date.
            ref = reference_date or date.today()
            season_name = self.season_resolver.season_name(ref)

        cache_key = None
        if use_cache and self.config.cache.enabled:
            resolved_year = year or 0
            cache_key = self.cache.observation_key(lon, lat, resolved_year, season_name)
            cached = self.cache.get(cache_key)
            if cached is not None:
                try:
                    return AgriculturalObservation.model_validate(cached)
                except Exception:  # noqa: BLE001 - stale/corrupt cache entry
                    self.cache.delete(cache_key)

        # -- Spatial -------------------------------------------------------- #
        location_info = self.matcher.location_info(lon, lat)

        # -- Temporal ------------------------------------------------------- #
        context: TemporalContext = self.matcher.resolve_temporal(
            year=year, season=season_name, reference_date=reference_date
        )

        # -- Tabular -------------------------------------------------------- #
        raw_name = (
            location_info.admin.village
            if location_info.admin and location_info.admin.village
            else location_info.dataset_location_name
        )
        location_resolution = resolve_location(raw_name)
        normalized_name = location_resolution.normalized
        taluk = normalize_name(location_info.admin.taluk) if location_info.admin else None
        district = (
            normalize_name(location_info.admin.district)
            if location_info.admin
            else None
        )
        # District-name alias: a boundary-derived admin name (e.g. taluk point
        # ``Bantwal`` inside district ``Dakshina Kannada``) is converted to the
        # record table's ``Location`` vocabulary (``Mangalore``) before the
        # village-level join, so queries whose nearest point is a taluk/district
        # centroid still resolve against the district-less data_season table
        # (see DISTRICT_TO_CSV). ``admin.village`` (most specific) is tried
        # first, then ``admin.district``. The ``district``/``taluk`` parameters
        # stay boundary-canonical: the ICRISAT fallback table matches on those.
        if location_info.admin is not None:
            for alias_source in (
                location_info.admin.village,
                location_info.admin.district,
            ):
                csv_name = district_to_csv(alias_source)
                if csv_name is not None:
                    normalized_name = csv_name
                    break
        tabular = self.matcher.match_tabular(
            village=normalized_name,
            taluk=taluk,
            district=district,
            year=context.year,
            season=context.season.name if context.season else None,
        )

        # -- Images (sequence) ---------------------------------------------- #
        result = self._build_sequence_with_fallback(
            lon, lat, context=context, resolution=resolution
        )

        temporal_info = TemporalInfo(
            year=context.year,
            season=context.season.name if context.season else None,
            season_months=(
                (context.season.start.month, context.season.end.month)
                if context.season
                else None
            ),
            observation_dates=result.sequence.sorted_dates,
            planting_start=context.planting_start,
            harvest_end=context.harvest_end,
            tolerance_days=self.config.temporal.tolerance_days,
        )

        # -- Quality -------------------------------------------------------- #
        quality = assess_quality(
            config=self.config.quality,
            location=location_info,
            temporal=temporal_info,
            tabular=tabular,
            sequence=result.sequence,
            distance_threshold_km=self.config.spatial.distance_threshold_km,
            additional_issues=result.issues,
        )

        # -- Historical context (same location + season across all years) --- #
        resolved_season_name = context.season.name if context.season else season_name
        historical_context = self.historical_context_builder.build(
            season_name=resolved_season_name,
            resolved_year=context.year,
            resolution=resolution,
        )

        observation = AgriculturalObservation(
            location=location_info,
            temporal=temporal_info,
            tabular=tabular or __empty_tabular__(),
            sequence=result.sequence,
            quality=quality,
            crop=tabular.crop if tabular else None,
            yield_value=tabular.yield_value if tabular else None,
            patch_size=self.config.patch.size,
            historical_context=historical_context,
            dataset_version=historical_context.dataset_version,
            season_calendar_version=historical_context.season_calendar_version,
            provenance={
                "stam_config": self.config.model_dump_json(),
                "ndvi_count": result.ndvi_count,
                "evi_count": result.evi_count,
                "paired_count": result.paired_count,
                "duplicate_dates": result.duplicate_dates,
                "cached": False,
                "season_resolver_source": self.season_resolver.source,
                "season_resolver_version": self.season_resolver.version,
                "historical_context_years": historical_context.years,
                "location_resolution": {
                    "raw_name": location_resolution.name,
                    "normalized_name": location_resolution.normalized,
                    "status": location_resolution.status,
                },
                "tabular_village": normalized_name,
            },
        )

        if use_cache and self.config.cache.enabled and cache_key is not None:
            self.cache.set(cache_key, observation.model_dump(mode="json"))

        logger.info(
            "Observation built",
            extra={
                "lon": lon, "lat": lat, "year": context.year,
                "season": context.season.name if context.season else None,
                "score": quality.overall_score,
            },
        )
        return observation

    def find_nearest(
        self, lon: float, lat: float, *, max_radius_km: float | None = None
    ) -> dict[str, Any]:
        """Nearest dataset location to a point (for map snapping / validation).

        Returns a serialisable dict with the matched point and distance.
        """
        self._require_initialized()
        match = self.matcher.find_nearest(lon, lat, max_radius_km=max_radius_km)
        return {
            "id": match.point.id,
            "name": match.point.name,
            "lon": match.point.lon,
            "lat": match.point.lat,
            "distance_km": round(match.distance_km, 4),
            "meta": match.point.meta,
        }

    def build_sequence(
        self,
        lon: float,
        lat: float,
        *,
        year: int,
        season: str | None = None,
        reference_date: date | None = None,
        resolution: str | None = None,
    ) -> SequenceBuildResult:
        """Build the ordered NDVI/EVI sequence for a point and season.

        Does not assemble the full observation (no tabular/quality) — useful
        for inspection and research workflows.
        """
        self._require_initialized()
        context = self.matcher.resolve_temporal(
            year=year, season=season, reference_date=reference_date
        )
        ndvi, evi = self.matcher.match_images(
            year=context.year, season=context.season, resolution=resolution
        )
        ndvi, evi = self._filter_images_for_point(lon, lat, ndvi, evi)
        return self.sequence_builder.build(ndvi, evi, resolution=resolution)

    def resolve_sequence(
        self,
        lon: float,
        lat: float,
        *,
        year: int,
        season: str | None = None,
        reference_date: date | None = None,
        resolution: str | None = None,
    ) -> SequenceInfo:
        """NDVI/EVI sequence for a point — no spatial or tabular matching.

        This is the legacy season-window path (``imagery.mode == "season"``).
        For multi-season imagery windows use :meth:`resolve_sequence_windowed`.
        """
        self._require_initialized()
        context = self.matcher.resolve_temporal(
            year=year, season=season, reference_date=reference_date
        )
        return self._build_sequence_with_fallback(
            lon, lat, context=context, resolution=resolution
        ).sequence

    def resolve_sequence_windowed(
        self,
        lon: float,
        lat: float,
        *,
        year: int,
        season: str | None = None,
        reference_date: date | None = None,
        resolution: str | None = None,
    ) -> SequenceInfo:
        """NDVI/EVI sequence resolved through a configurable imagery window.

        Mirrors :meth:`resolve_sequence` (never consults the spatial location
        index or tabular chain — safe for arbitrary GPS points) but acquires
        imagery via :func:`training.stam.temporal_window.resolve_window`
        instead of the season calendar window:

        * ``window_days`` / ``crop_year`` anchor on ``reference_date`` (the
          survey date) when provided;
        * records inside the window are verified to cover the point, paired by
          date, and trimmed to ``<= max_observations`` real frames by the
          configured strategy;
        * ``mode == "season"`` simply delegates to :meth:`resolve_sequence`.

        Every retained frame is a real record that exists on disk — no date is
        fabricated, duplicated or zero-filled here.
        """
        self._require_initialized()
        imagery = self.config.imagery
        if imagery.mode == "season":
            return self.resolve_sequence(
                lon,
                lat,
                year=year,
                season=season,
                reference_date=reference_date,
                resolution=resolution,
            )
        context = self.matcher.resolve_temporal(
            year=year, season=season, reference_date=reference_date
        )
        return self._build_sequence_windowed(
            lon,
            lat,
            context=context,
            reference_date=reference_date,
            resolution=resolution,
        ).sequence

    def get_patch(
        self,
        image_path: str,
        lon: float,
        lat: float,
        *,
        size: int | None = None,
        pad_mode: str | None = None,
        pad_value: float | None = None,
    ) -> RasterPatch:
        """Extract a fixed-size spatial patch centred on ``(lon, lat)``.

        The image path must be a Dataset Manager metadata record path.
        """
        self._require_initialized()
        return self.patch_generator.get_patch(
            image_path,
            lon,
            lat,
            size=size,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )

    def validate(self, observation: AgriculturalObservation) -> QualityReport:
        """Re-run the quality-control pass over an existing observation."""
        return assess_quality(
            config=self.config.quality,
            location=observation.location,
            temporal=observation.temporal,
            tabular=observation.tabular,
            sequence=observation.sequence,
            distance_threshold_km=self.config.spatial.distance_threshold_km,
        )

    def summary(self) -> dict[str, Any]:
        """Configuration + index statistics (for dashboards / CLI)."""
        return {
            "initialized": self._initialized,
            "patch_size": self.config.patch.size,
            "resolution": self.config.image.resolution,
            "max_search_radius_km": self.config.spatial.max_search_radius_km,
            "distance_threshold_km": self.config.spatial.distance_threshold_km,
            "seasons": [s.name for s in self.config.seasons],
            "indexes": self.matcher.spatial_stats() if self._initialized else {},
            "cache_enabled": self.config.cache.enabled,
            "season_resolver": {
                "source": self.season_resolver.source,
                "version": self.season_resolver.version,
            },
        }

    # -- Internals ------------------------------------------------------------ #

    def _build_sequence_with_fallback(
        self,
        lon: float,
        lat: float,
        *,
        context: TemporalContext,
        resolution: str | None,
    ) -> SequenceBuildResult:
        """Build the image sequence; degrade gracefully when strict pairing fails."""
        ndvi, evi = self.matcher.match_images(
            year=context.year, season=context.season, resolution=resolution
        )
        ndvi, evi = self._filter_images_for_point(lon, lat, ndvi, evi)
        try:
            return self.sequence_builder.build(ndvi, evi, resolution=resolution)
        except PairingError as exc:
            logger.warning(
                "Strict pairing failed; rebuilding lenient sequence",
                extra={"reason": str(exc)},
            )
            lenient = ObservationSequenceBuilder(
                require_pairs=False,
                max_gap_days=self.config.quality.max_temporal_gap_days,
            )
            result = lenient.build(ndvi, evi, resolution=resolution)
            result.issues.insert(
                0,
                _missing_pair_issue(),
            )
            return result

    def _build_sequence_windowed(
        self,
        lon: float,
        lat: float,
        *,
        context: TemporalContext,
        reference_date: date | None,
        resolution: str | None,
    ) -> SequenceBuildResult:
        """Build a sequence from imagery-window frames (never fabricates frames).

        Steps: resolve the acquisition window, collect windowed records,
        verify point coverage, pair by date, then trim the *real* frames to
        ``<= max_observations`` using the configured selection strategy. The
        trim only drops real records — any sequence slot that remains empty is
        handled downstream by the standard zero-fill + temporal-mask policy.
        """
        from .temporal_window import (
            resolve_window,
            select_temporal_frames,
            sequence_from_pairs,
            window_description,
        )

        imagery = self.config.imagery
        try:
            window = resolve_window(
                imagery,
                reference_date=reference_date,
                year=context.year,
                season=context.season,
            )
        except ValueError as exc:
            logger.warning(
                "Imagery window unresolved",
                extra={"error": str(exc), "lon": lon, "lat": lat},
            )
            raise

        ndvi, evi = self.matcher.match_images_in_window(
            window, year=context.year, resolution=resolution
        )
        ndvi, evi = self._filter_images_for_point(lon, lat, ndvi, evi)

        try:
            result = self.sequence_builder.build(ndvi, evi, resolution=resolution)
        except PairingError as exc:
            logger.warning(
                "Strict pairing failed; rebuilding lenient windowed sequence",
                extra={"reason": str(exc)},
            )
            lenient = ObservationSequenceBuilder(
                require_pairs=False,
                max_gap_days=self.config.quality.max_temporal_gap_days,
            )
            result = lenient.build(ndvi, evi, resolution=resolution)
            result.issues.insert(
                0,
                _missing_pair_issue(),
            )

        kept = select_temporal_frames(
            result.sequence.pairs, reference_date, imagery
        )
        sequence = sequence_from_pairs(result.sequence, kept)

        desc = window_description(imagery, reference_date)
        for pair in sequence.pairs:
            pair.quality["imagery_window"] = desc
        logger.info(
            "Windowed sequence built",
            extra={
                "window": desc,
                "candidate_frames": len(result.sequence.pairs),
                "kept_frames": len(sequence.pairs),
                "mode": imagery.mode,
                "strategy": imagery.strategy,
            },
        )
        return SequenceBuildResult(
            sequence=sequence,
            issues=result.issues,
            ndvi_count=result.ndvi_count,
            evi_count=result.evi_count,
            paired_count=result.paired_count,
            duplicate_dates=result.duplicate_dates,
        )

    def _filter_images_for_point(
        self,
        lon: float,
        lat: float,
        ndvi: list[Any],
        evi: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Drop image records whose raster does not cover the query point.

        ``match_images`` returns every season-window raster in the region, but a
        location almost never falls inside all of them. Reading a patch from a
        non-covering raster raises ``PatchOutOfBoundsError`` at training time, so
        records are kept only when the point (projected into the raster's CRS)
        is inside the raster bounds expanded by half a patch — partial edge
        patches still pass through to the edge-correction/padding logic.
        """
        records = [*ndvi, *evi]
        margin = _half_patch_units(self.config.patch.size, records)
        return (
            [r for r in ndvi if _record_covers_point(r, lon, lat, margin)],
            [r for r in evi if _record_covers_point(r, lon, lat, margin)],
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _record_covers_point(record, lon: float, lat: float, margin_units: float) -> bool:
    """True when ``(lon, lat)`` falls inside ``record``'s raster bounds.

    ``record.bounds`` is expressed in the raster's native CRS, so the point is
    projected into that CRS before the comparison. Records without usable
    bounds or CRS are kept (best effort) so behaviour is unchanged for them.
    """
    if not getattr(record, "bounds", None) or not getattr(record, "crs", None):
        return True
    try:
        x, y = transform_point(WGS84, record.crs, lon, lat)
    except Exception:  # noqa: BLE001 - keep the record on projection failure
        return True
    left, bottom, right, top = record.bounds
    return (
        left - margin_units <= x <= right + margin_units
        and bottom - margin_units <= y <= top + margin_units
    )


def _half_patch_units(size: int, records: list[Any]) -> float:
    """Half a patch in raster units (max pixel size across the records)."""
    half_pixels = size // 2
    max_pixel = 0.0
    for record in records:
        px = getattr(record, "pixel_size", None)
        if px:
            max_pixel = max(max_pixel, float(px[0]), float(px[1]))
    return half_pixels * max_pixel


def _missing_pair_issue():
    from .observation import QualityIssue

    return QualityIssue(
        code="ST-Q-PAIR-002",
        severity="error",
        message="Strict NDVI/EVI pairing failed for one or more dates",
    )


def __empty_tabular__():
    """An empty tabular-features placeholder (kept private)."""
    from .observation import TabularFeatures

    return TabularFeatures(crop=None, yield_value=None, fields={}, matched_level="none")
