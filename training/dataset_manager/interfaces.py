"""Ports (abstract interfaces) for the Dataset Manager.

The Dataset Manager follows a clean / hexagonal architecture: every
capability is declared as an abstract interface in this module and the
concrete implementations live in their own modules. :class:`DatasetManager`
depends only on these ports (dependency inversion), so tests and future
alternatives can inject fakes without touching the orchestration layer.

The following ports are declared:

* :class:`Downloader`            — acquire the primary Kaggle dataset.
* :class:`Scanner`               — discover files and build an inventory.
* :class:`Validator`             — validate structure / integrity.
* :class:`MetadataGenerator`     — build :class:`MetadataRecord` objects.
* :class:`MetadataStore`         — persist and query metadata records.
* :class:`CSVLoader`             — discover, profile and load CSV files.
* :class:`ImageLoader`           — lazy GeoTIFF metadata / windowed reads.
* :class:`Cache`                 — generic key/value caching.
* :class:`Registry`              — dataset lifecycle registry.
* :class:`VersionManager`        — semantic versioning of datasets.
* :class:`ProviderRegistry`      — provider registration / resolution / health.
* :class:`SpatialIndex`          — village/district/coordinate lookups.
* :class:`PatchExtractor`        — geographic patch extraction from rasters.
* :class:`HistoricalContextBuilder` — multi-year per-location observation context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .models import (
    CSVProfile,
    DatasetInventory,
    DatasetStatus,
    MetadataRecord,
    RasterMetadata,
    SpatialMetadata,
    SpatialRecord,
    ValidationReport,
    VersionEntry,
)


class Downloader(ABC):
    """Acquire the primary Kaggle dataset."""

    @abstractmethod
    def download(self, handle: str, *, force: bool = False) -> Path:
        """Ensure the dataset is present locally and return its root path.

        Should detect an existing download and skip the network transfer when
        possible. ``force=True`` triggers a fresh download.
        """

    @abstractmethod
    def is_downloaded(self, handle: str) -> bool:
        """True when the dataset already exists in the local cache."""

    @abstractmethod
    def materialize(self, source: Path, destination: Path) -> int:
        """Copy/hard-link files from ``source`` into ``destination``.

        Returns the number of files materialised.
        """


class Scanner(ABC):
    """Discover files under a dataset root and classify them."""

    @abstractmethod
    def scan(self, root: Path, *, use_cache: bool = True) -> DatasetInventory:
        """Produce an inventory of every file under ``root``.

        The result may be served from the cache when the tree signature is
        unchanged and ``use_cache`` is true.
        """


class Validator(ABC):
    """Validate folder structure, integrity and metadata completeness."""

    @abstractmethod
    def validate(self, root: Path, inventory: DatasetInventory) -> ValidationReport:
        """Run all validation checks and return a detailed report."""


class MetadataGenerator(ABC):
    """Produce one :class:`MetadataRecord` per scanned file."""

    @abstractmethod
    def generate(
        self, root: Path, inventory: DatasetInventory, *, force: bool = False
    ) -> list[MetadataRecord]:
        """Generate metadata for every file in ``inventory``.

        ``force=True`` regenerates even when a record already exists.
        """


class MetadataStore(ABC):
    """Persist and query metadata records."""

    @abstractmethod
    def save(self, record: MetadataRecord) -> None:
        """Insert or replace the record (upsert on relative path)."""

    @abstractmethod
    def save_many(self, records: list[MetadataRecord]) -> int:
        """Bulk upsert; returns the number of rows written."""

    @abstractmethod
    def get(self, path: Path) -> MetadataRecord | None:
        """Fetch the record for a file path, if present."""

    @abstractmethod
    def query(self, **filters: Any) -> list[MetadataRecord]:
        """Query records by any of: year, index_type, resolution, category."""

    @abstractmethod
    def count(self) -> int:
        """Total number of metadata records."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (no-op for per-call connections)."""

    @abstractmethod
    def export_parquet(self, path: Path) -> Path:
        """Export the full metadata table to a Parquet file."""


class CSVLoader(ABC):
    """Discover, profile and load tabular (CSV) datasets."""

    @abstractmethod
    def discover(self, root: Path) -> list[Path]:
        """Recursively find every CSV file under ``root``."""

    @abstractmethod
    def infer_schema(self, path: Path) -> CSVProfile:
        """Infer columns, dtypes and missing-value counts for ``path``."""

    @abstractmethod
    def preview(self, path: Path, n_rows: int = 5) -> Any:
        """Return the first ``n_rows`` rows as a data frame / records."""

    @abstractmethod
    def load(self, path: Path, *, chunksize: int | None = None, **kwargs: Any) -> Any:
        """Load a CSV into a data frame (or an iterator when chunked)."""


class ImageLoader(ABC):
    """Lazy, header-only access to GeoTIFF rasters."""

    @abstractmethod
    def read_metadata(self, path: Path) -> RasterMetadata:
        """Read header metadata without loading raster data."""

    @abstractmethod
    def preview(self, path: Path) -> dict[str, Any]:
        """Dimensions, dtype and a small sampled statistic block."""

    @abstractmethod
    def read_window(self, path: Path, *, window: tuple[int, int, int, int], band: int = 1) -> Any:
        """Read a bounded window of a band (``(row_off, col_off, height, width)``)."""


class Cache(ABC):
    """Key/value cache with TTL and prefix invalidation."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the deserialised value for ``key``, or None on miss/expiry."""

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove ``key``; returns True when it existed."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Remove all keys starting with ``prefix``; returns the count."""

    @abstractmethod
    def clear(self) -> int:
        """Drop all entries; returns the count removed."""

    @abstractmethod
    def prune(self) -> int:
        """Remove expired entries; returns the count removed."""


class Registry(ABC):
    """Dataset lifecycle registry."""

    @abstractmethod
    def register(
        self,
        *,
        name: str,
        source: str,
        root_path: Path,
        version: str = "0.0.0",
        status: DatasetStatus = DatasetStatus.PENDING,
    ) -> int:
        """Create a dataset entry and return its id."""

    @abstractmethod
    def get(self, dataset_id: int) -> dict[str, Any] | None:
        """Fetch an entry by id."""

    @abstractmethod
    def get_by_name(self, name: str) -> dict[str, Any] | None:
        """Fetch the most recent entry with the given name."""

    @abstractmethod
    def list(self) -> list[dict[str, Any]]:
        """List all registered datasets, newest first."""

    @abstractmethod
    def update_status(self, dataset_id: int, status: DatasetStatus) -> None:
        """Update the lifecycle status of an entry."""

    @abstractmethod
    def update_checksum(self, dataset_id: int, checksum: str, file_count: int) -> None:
        """Record an integrity checksum and file count."""

    @abstractmethod
    def remove(self, dataset_id: int) -> bool:
        """Remove an entry (and its versions); returns True when removed."""


class VersionManager(ABC):
    """Semantic versioning of a registered dataset."""

    @abstractmethod
    def current(self, dataset_id: int) -> str | None:
        """Return the current version string, or None."""

    @abstractmethod
    def list(self, dataset_id: int) -> list[VersionEntry]:
        """Return version history, newest first."""

    @abstractmethod
    def bump(
        self, dataset_id: int, part: str = "patch", *, message: str = ""
    ) -> VersionEntry:
        """Bump the current version (major/minor/patch) and snapshot it."""

    @abstractmethod
    def rollback(self, dataset_id: int, version: str) -> VersionEntry:
        """Restore ``version`` as the current version."""

    @abstractmethod
    def snapshot(
        self,
        dataset_id: int,
        version: str,
        *,
        message: str,
        checksum: str | None,
        file_count: int,
    ) -> VersionEntry:
        """Record a version snapshot (used by bump/rollback)."""


class ProviderRegistry(ABC):
    """Registry of named data providers.

    The Dataset Manager **never constructs providers directly** — it resolves
    them through this registry, which owns registration, health checks,
    capability queries, priority ordering, availability and discovery (so
    future plugins can be added purely from configuration).
    """

    @abstractmethod
    def register(
        self,
        name: str,
        kind: str,
        provider: Any,
        *,
        enabled: bool = True,
        priority: int = 100,
        config: dict[str, Any] | None = None,
    ) -> Any:
        """Register a provider instance under ``name``."""

    @abstractmethod
    def resolve(self, name: str) -> Any:
        """Return the live provider instance for ``name``.

        Raises :class:`~training.dataset_manager.exceptions.DatasetNotFoundError`
        when the provider is unknown or disabled.
        """

    @abstractmethod
    def resolve_by_kind(self, kind: str) -> list[Any]:
        """Providers of a kind, sorted by priority (highest first)."""

    @abstractmethod
    def names(self) -> list[str]:
        """Registered provider names (sorted)."""

    @abstractmethod
    def has(self, name: str) -> bool:
        """True when a provider is registered under ``name``."""

    @abstractmethod
    def priority(self, name: str) -> int:
        """Resolution priority of a registered provider."""

    @abstractmethod
    def availability(self) -> dict[str, bool]:
        """``{name: available}`` for every registered provider."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """``{name: capabilities}`` for every registered provider."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """``{name: health_snapshot}`` for every registered provider."""

    @abstractmethod
    def discovery(self) -> list[dict[str, Any]]:
        """Plain registration records (for diagnostics / reports)."""


class SpatialIndex(ABC):
    """Spatial lookup over named locations (villages / districts / ...).

    Backed by a KD-tree over WGS84 coordinates for nearest-neighbour and
    radius searches, plus name indexes for village/district lookups.
    """

    @abstractmethod
    def build(self, records: list[SpatialRecord]) -> int:
        """(Re)build the index from ``records``; returns the count."""

    @abstractmethod
    def lookup_village(self, name: str) -> list[SpatialRecord]:
        """Locations of kind ``village`` whose name matches ``name``."""

    @abstractmethod
    def lookup_district(self, name: str) -> list[SpatialRecord]:
        """Locations of kind ``district`` whose name matches ``name``."""

    @abstractmethod
    def records(self) -> list[SpatialRecord]:
        """All indexed records."""

    @abstractmethod
    def nearest(
        self, latitude: float, longitude: float, k: int = 1
    ) -> list[tuple[SpatialRecord, float]]:
        """``k`` nearest records to a point as ``(record, distance_deg)``."""

    @abstractmethod
    def search_coordinates(
        self, latitude: float, longitude: float, tolerance: float = 0.01
    ) -> list[SpatialRecord]:
        """Records within ``tolerance`` degrees of a point (exact match)."""

    @abstractmethod
    def within_bbox(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> list[SpatialRecord]:
        """Records inside a longitude/latitude bounding box."""

    @abstractmethod
    def within_radius(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[SpatialRecord]:
        """Records within ``radius_km`` kilometres of a point."""

    @abstractmethod
    def metadata(self) -> SpatialMetadata:
        """Aggregate spatial metadata (counts, bounds, timestamp)."""


class PatchExtractor(ABC):
    """Geographic patch extraction from imagery.

    Given a latitude / longitude (WGS84) the extractor locates the best
    matching raster (by index type, resolution, year and proximity), converts
    the point into the raster's CRS, extracts a square window and optionally
    pads it to the exact requested size. Returns raw NumPy arrays — no raster
    preprocessing is applied.
    """

    @abstractmethod
    def extract(
        self,
        latitude: float,
        longitude: float,
        size: int,
        *,
        index_type: str | None = None,
        resolution: str | None = None,
        year: int | None = None,
        band: int = 1,
        padding: bool = True,
    ) -> Any:
        """Extract a ``size`` x ``size`` patch around (lat, lon)."""


class HistoricalContextBuilder(ABC):
    """Build the multi-year, per-location observation context.

    Combines the tabular record for a location with the satellite records
    (NDVI / EVI metadata, observation dates, resolutions) for every available
    year. **No STAM inference is executed** — this is raw context only.
    """

    @abstractmethod
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
    ) -> Any:
        """Return the :class:`HistoricalObservationSet` for a location."""
