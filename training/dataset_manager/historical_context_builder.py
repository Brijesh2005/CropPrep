"""Multi-year, per-location observation context builder.

:class:`HistoricalContextBuilder` answers the question the models ask before
inference: *"everything we know about this location across every available
year"* — the tabular record plus the NDVI / EVI satellite records, their
observation dates and resolution bands.

It composes existing Dataset Manager capabilities (tabular provider discovery,
metadata store queries, spatial index lookups) and returns a
:class:`HistoricalObservationSet`. **STAM is intentionally not executed here**
— this layer gathers raw context; inference happens later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .exceptions import DatasetNotFoundError
from .interfaces import (
    HistoricalContextBuilder,
    MetadataStore,
    SpatialIndex,
)
from .logger import get_logger
from .metadata_repository import MetadataRepository
from .models import (
    FileCategory,
    HistoricalObservation,
    HistoricalObservationSet,
    IndexType,
)
from .providers.base import ImageProvider, TabularProvider

logger = get_logger("historical_context_builder")

#: Column name fragments that identify a location key in a tabular dataset.
_LOCATION_COLUMN_FRAGMENTS = (
    "village", "district", "taluk", "tehsil", "block", "state", "mandal",
)
#: Column name fragments that identify a year key in a tabular dataset.
_YEAR_COLUMN_NAMES = ("year", "yr", "season_year", "crop_year", "year_of")


class HistoricalContextBuilderImpl(HistoricalContextBuilder):
    """Concrete :class:`HistoricalContextBuilder`.

    Args:
        tabular_provider: Optional :class:`TabularProvider` for location-keyed
            records (None skips the tabular part).
        image_provider: Optional :class:`ImageProvider` for satellite records
            (None skips the image part).
        metadata_store: Optional :class:`MetadataStore` for persisted raster
            metadata (falls back to the image provider catalog).
        spatial_index: Optional :class:`SpatialIndex` used to resolve
            village / district names to coordinates.
        metadata_repository: Optional :class:`MetadataRepository` used to
            persist per-year temporal records.
        max_frame_rows: Upper bound of rows scanned per tabular dataset.
    """

    def __init__(
        self,
        *,
        tabular_provider: TabularProvider | None = None,
        image_provider: ImageProvider | None = None,
        metadata_store: MetadataStore | None = None,
        spatial_index: SpatialIndex | None = None,
        metadata_repository: MetadataRepository | None = None,
        max_frame_rows: int = 200_000,
    ) -> None:
        self.tabular_provider = tabular_provider
        self.image_provider = image_provider
        self.metadata_store = metadata_store
        self.spatial_index = spatial_index
        self.metadata_repository = metadata_repository
        self.max_frame_rows = max_frame_rows

    # -- Public API ------------------------------------------------------------ #

    def build(
        self,
        *,
        village: str | None = None,
        district: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        index_type: str | None = None,
        resolution: str | None = None,
        years: list[int] | None = None,
    ) -> HistoricalObservationSet:
        """Build all observations for one location across every available year.

        Args:
            village / district: Location name resolved through the spatial
                index (preferred).
            latitude / longitude: Direct WGS84 coordinates (used when no name
                is given, or to locate imagery when no spatial record exists).
            index_type: Restrict satellite records (``"NDVI"`` / ``"EVI"``).
            resolution: Restrict to ``"R10m"`` / ``"R20m"``.
            years: Restrict to specific years.

        Returns:
            A :class:`HistoricalObservationSet`.

        Raises:
            DatasetNotFoundError: When neither a name nor coordinates are given.
        """
        label, lat, lon = self._resolve_location(
            village=village, district=district, latitude=latitude, longitude=longitude
        )

        wanted_index = _normalise_index(index_type)
        available_years = self._available_years(wanted_index, resolution)
        target_years = years or available_years
        target_years = sorted(set(target_years))

        tabular_by_year = self._match_tabular(label)
        observations: list[HistoricalObservation] = []
        present_years: set[int] = set()

        for year in target_years:
            ndvi, evi = self._image_records(year, wanted_index, resolution)
            if not ndvi and not evi and year not in tabular_by_year:
                continue
            present_years.add(year)
            observation_dates = sorted(
                {r.get("observation_date") for r in (ndvi + evi) if r.get("observation_date")}
            )
            quality = {
                "records": len(ndvi) + len(evi),
                "ndvi": len(ndvi),
                "evi": len(evi),
                "has_tabular": year in tabular_by_year,
                "resolutions": sorted(
                    {r.get("resolution") for r in (ndvi + evi) if r.get("resolution")}
                ),
            }
            if wanted_index is not None:
                present = len(ndvi) if wanted_index is IndexType.NDVI else len(evi)
                quality["missing_index"] = present == 0
            observations.append(
                HistoricalObservation(
                    year=year,
                    tabular=tabular_by_year.get(year),
                    tabular_source=self._tabular_source if year in tabular_by_year else None,
                    ndvi=ndvi,
                    evi=evi,
                    observation_dates=observation_dates,
                    quality=quality,
                )
            )

        years_sorted = sorted(present_years)
        missing_years = self._missing_years(years_sorted, target_years)

        overall_quality = {
            "years_covered": len(years_sorted),
            "years_observed": len(target_years),
            "missing_years": missing_years,
            "total_records": sum(len(o.ndvi) + len(o.evi) for o in observations),
            "has_tabular": any(o.tabular is not None for o in observations),
            "resolution": resolution,
            "index_type": index_type,
        }
        result = HistoricalObservationSet(
            location=label,
            latitude=lat,
            longitude=lon,
            years=years_sorted,
            observations=observations,
            missing_years=missing_years,
            quality=overall_quality,
        )

        self._persist_temporal(result)
        return result

    # -- Location resolution --------------------------------------------------- #

    def _resolve_location(
        self,
        *,
        village: str | None,
        district: str | None,
        latitude: float | None,
        longitude: float | None,
    ) -> tuple[str, float | None, float | None]:
        if village or district:
            label = village or district
            if self.spatial_index is not None:
                records = []
                if village:
                    records = self.spatial_index.lookup_village(village)
                elif district:
                    records = self.spatial_index.lookup_district(district)
                if records:
                    wanted_kind = "village" if village else "district"
                    exact = [r for r in records if r.kind == wanted_kind]
                    record = (exact or records)[0]
                    return record.name, record.latitude, record.longitude
                if latitude is not None and longitude is not None:
                    return label, float(latitude), float(longitude)
                if district is not None:
                    # No district-kind record — keep the requested name; the
                    # satellite records are gathered regardless of coordinates.
                    return label, None, None
                raise DatasetNotFoundError(
                    f"Could not resolve location {label!r} in the spatial index"
                )
            if latitude is not None and longitude is not None:
                return label, float(latitude), float(longitude)
            raise DatasetNotFoundError(
                f"Could not resolve location {label!r} in the spatial index"
            )
        if latitude is not None and longitude is not None:
            return f"{latitude}, {longitude}", float(latitude), float(longitude)
        raise DatasetNotFoundError(
            "Historical context requires a village/district name or coordinates"
        )

    # -- Year / record gathering ------------------------------------------------ #

    def _available_years(
        self, index: IndexType | None, resolution: str | None
    ) -> list[int]:
        if self.image_provider is None:
            return []
        catalog = self.image_provider.catalog()
        years = set(catalog.years)
        if index is not None:
            entries = catalog.ndvi if index is IndexType.NDVI else catalog.evi
            years = {e.year for e in entries if e.year is not None}
        if resolution:
            wanted = resolution.upper().replace(" ", "")
            entries = (
                catalog.ndvi + catalog.evi
            )
            years = {
                e.year for e in entries
                if e.year is not None
                and e.resolution.value.upper() == wanted
            }
        return sorted(years)

    def _image_records(
        self, year: int, index: IndexType | None, resolution: str | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.image_provider is None:
            return [], []
        records: list[dict[str, Any]] = []
        if self.metadata_store is not None:
            try:
                stored = self.metadata_store.query(category="geotiff", year=year)
                for record in stored:
                    if record.observation_date is None:
                        continue
                    if index is not None and record.index_type is not index:
                        continue
                    if resolution and record.resolution.value.upper() != resolution.upper():
                        continue
                    records.append(
                        {
                            "path": str(record.path),
                            "relative_path": record.relative_path,
                            "index_type": record.index_type.value,
                            "resolution": record.resolution.value,
                            "observation_date": record.observation_date.isoformat(),
                            "year": record.year,
                        }
                    )
            except Exception:  # noqa: BLE001 - fall back to the catalog
                records = []
        if not records:
            catalog = self.image_provider.catalog()
            entries = [e for e in catalog.entries if e.year == year]
            from .utils import parse_observation_date

            for entry in entries:
                if entry.category is not FileCategory.GEOTIFF:
                    continue
                if index is not None and entry.index_type is not index:
                    continue
                if resolution and entry.resolution.value.upper() != resolution.upper():
                    continue
                obs_date = parse_observation_date(entry.path)
                if obs_date is None:
                    continue
                records.append(
                    {
                        "path": str(entry.path),
                        "relative_path": entry.relative_path,
                        "index_type": entry.index_type.value,
                        "resolution": entry.resolution.value,
                        "observation_date": obs_date.isoformat(),
                        "year": entry.year,
                    }
                )
        ndvi = [r for r in records if r["index_type"] == "NDVI"]
        evi = [r for r in records if r["index_type"] == "EVI"]
        return ndvi, evi

    def _missing_years(
        self, present: list[int], target: list[int]
    ) -> list[int]:
        if not target:
            return []
        present_set = set(present)
        if len(target) >= 2:
            lo, hi = target[0], target[-1]
            span = [y for y in range(lo, hi + 1)]
        else:
            span = target
        return sorted(set(span) - present_set)

    # -- Tabular matching ------------------------------------------------------ #

    def _match_tabular(self, label: str) -> dict[int, dict[str, Any]]:
        """Find per-year tabular rows for a location (best effort).

        Scans discovered tabular datasets for a location column, filters rows
        matching ``label`` (case-insensitive), and groups them by a year
        column when one exists. Datasets without a location column are skipped
        after the schema check.
        """
        self._tabular_source: str | None = None
        if self.tabular_provider is None:
            return {}

        label_lower = str(label).strip().lower()
        for name in self.tabular_provider.names():
            try:
                schema = self.tabular_provider.schema(name)
            except Exception:  # noqa: BLE001 - best effort
                continue
            columns = list(schema.get("columns", []))
            location_col = _find_location_column(columns)
            if location_col is None:
                continue
            try:
                frame = self.tabular_provider.load(name)
            except Exception:  # noqa: BLE001 - best effort
                continue
            if len(frame) > self.max_frame_rows:
                frame = frame.head(self.max_frame_rows)
            matched = frame[
                frame[location_col].astype(str).str.strip().str.lower() == label_lower
            ]
            if matched.empty:
                continue
            year_col = _find_year_column(matched.columns.tolist())
            grouped: dict[int, dict[str, Any]] = {}
            if year_col is None:
                grouped[0] = _row_to_dict(matched.iloc[0])
            else:
                for year, group in matched.groupby(year_col):
                    year_value = _coerce_year(year)
                    if year_value is None:
                        continue
                    grouped[year_value] = _row_to_dict(group.iloc[0])
            if grouped:
                self._tabular_source = name
                return grouped
        return {}

    def _persist_temporal(self, result: HistoricalObservationSet) -> None:
        """Persist per-year temporal availability records (best effort)."""
        if self.metadata_repository is None:
            return
        from .models import TemporalRecord

        for observation in result.observations:
            for index_type in ("NDVI", "EVI"):
                records = observation.ndvi if index_type == "NDVI" else observation.evi
                if not records:
                    continue
                resolutions = sorted({r["resolution"] for r in records})
                for resolution in resolutions:
                    res_records = [r for r in records if r["resolution"] == resolution]
                    self.metadata_repository.save_temporal(
                        TemporalRecord(
                            index_type=index_type,
                            year=observation.year,
                            resolution=resolution,
                            count=len(res_records),
                            observation_months=[
                                int(d[5:7]) for d in observation.observation_dates if d
                            ],
                            observation_dates=observation.observation_dates,
                        )
                    )


def _normalise_index(index_type: str | None) -> IndexType | None:
    if index_type is None:
        return None
    upper = str(index_type).upper()
    if upper == "NDVI":
        return IndexType.NDVI
    if upper == "EVI":
        return IndexType.EVI
    raise ValueError(f"Unsupported index type: {index_type} (expected NDVI/EVI)")


def _find_location_column(columns: list[str]) -> str | None:
    lowered = {c: c.lower() for c in columns}
    for column in columns:
        name = lowered[column]
        if any(fragment in name for fragment in _LOCATION_COLUMN_FRAGMENTS):
            return column
    return None


def _find_year_column(columns: list[str]) -> str | None:
    lowered = {c: c.lower() for c in columns}
    for column in columns:
        if lowered[column] in _YEAR_COLUMN_NAMES:
            return column
    return None


def _coerce_year(value: Any) -> int | None:
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def _row_to_dict(row: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in row.index:
        out[str(column)] = str(row[column])
    return out
