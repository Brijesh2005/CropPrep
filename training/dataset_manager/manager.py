"""The :class:`DatasetManager` facade — the public API of the DMS.

The manager is the **single entry point** every other module (AI, GIS,
backend, frontend) uses to access datasets. It wires the concrete components
together via dependency injection and exposes a small, stable surface:

* :meth:`download` / :meth:`scan` / :meth:`validate` /
  :meth:`generate_metadata` — the pipeline steps.
* :meth:`inventory` / :meth:`summary` — discovery results.
* :meth:`load_csv` / :meth:`load_image` — **the only** ways to read data.
* :meth:`get_metadata` / :meth:`query_metadata` /
  :meth:`export_metadata_parquet` — metadata access.
* :meth:`register` / :meth:`bump_version` / :meth:`rollback_version` —
  lifecycle + versioning.
* :meth:`cache_get` / :meth:`cache_set` / :meth:`cache_invalidate` —
  cache control.

Construct with :meth:`DatasetManager.from_config` (factory) or by passing a
:class:`Settings` object. Individual components may be injected for tests.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .cache_manager import CacheManager
from .config import Settings, load_settings
from .csv_loader import PandasCSVLoader
from .dataset_registry import SQLiteRegistry
from .downloader import KaggleDownloader
from .exceptions import DatasetNotFoundError
from .image_loader import RasterioImageLoader
from .interfaces import (
    Cache,
    CSVLoader,
    Downloader,
    ImageLoader,
    MetadataGenerator,
    MetadataStore,
    Registry,
    Scanner,
    Validator,
    VersionManager,
)
from .logger import get_logger, setup_logging
from .manager_paths import ensure_state_dirs
from .metadata import MetadataGeneratorImpl, SQLiteMetadataStore
from .models import (
    DatasetInventory,
    DatasetStatus,
    DatasetSummary,
    HistoricalContext,
    IndexType,
    MetadataRecord,
    Resolution,
    ValidationReport,
    VersionEntry,
)
from .providers import GitRepositoryTabularProvider, KaggleHubImageProvider
from .providers.models import (
    ImageCatalog,
    ImageDatasetLocation,
    PatchRequest,
    TabularCatalog,
    TabularJoinSpec,
)
from .scanner import DatasetScanner
from .validator import DatasetValidator
from .version_manager import SQLiteVersionManager

logger = get_logger("manager")

__all__ = ["DatasetManager"]


class DatasetManager:
    """Facade over all Dataset Manager capabilities.

    Args:
        settings: Validated :class:`Settings`.
        downloader: Optional :class:`Downloader` (default: Kaggle).
        scanner: Optional :class:`Scanner` (default: parallel scanner).
        validator: Optional :class:`Validator`.
        csv_loader: Optional :class:`CSVLoader`.
        image_loader: Optional :class:`ImageLoader`.
        metadata_generator: Optional :class:`MetadataGenerator`.
        metadata_store: Optional :class:`MetadataStore`.
        registry: Optional :class:`Registry`.
        version_manager: Optional :class:`VersionManager`.
        cache: Optional :class:`Cache`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        downloader: Downloader | None = None,
        scanner: Scanner | None = None,
        validator: Validator | None = None,
        csv_loader: CSVLoader | None = None,
        image_loader: ImageLoader | None = None,
        metadata_generator: MetadataGenerator | None = None,
        metadata_store: MetadataStore | None = None,
        registry: Registry | None = None,
        version_manager: VersionManager | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._configure_logging()

        ensure_state_dirs(self.settings)

        # -- Dependencies (inject or build with the default factory) ----------- #
        self.cache = cache or CacheManager(
            self.settings.cache_db_path(),
            config=self.settings.cache,
        )
        self.downloader = downloader or KaggleDownloader(self.settings.download)
        self.scanner = scanner or DatasetScanner(self.settings.scan, cache=self.cache)
        self.csv_loader = csv_loader or PandasCSVLoader(self.settings.scan)
        self.image_loader = image_loader or RasterioImageLoader()
        self.metadata_store = metadata_store or SQLiteMetadataStore(
            self.settings.metadata_db_path()
        )
        self.registry = registry or SQLiteRegistry(self.settings.registry_db_path())
        self.version_manager = version_manager or SQLiteVersionManager(
            self.settings.registry_db_path(), self.registry
        )
        self.metadata_generator = metadata_generator or MetadataGeneratorImpl(
            self.settings.metadata,
            csv_loader=self.csv_loader,
            image_loader=self.image_loader,
            store=self.metadata_store,
        )
        self.validator = validator or DatasetValidator(
            self.settings.validation,
            image_loader=self.image_loader,
            metadata_store=self.metadata_store,
        )

        # -- Providers (the only way data sources are touched) ------------------- #
        tab_cfg = self.settings.providers.tabular
        self.tabular_provider = GitRepositoryTabularProvider(
            root=self.settings.tabular_root,
            loader=self.csv_loader,
            patterns=list(tab_cfg.patterns),
        )
        img_cfg = self.settings.providers.image
        self.image_provider = KaggleHubImageProvider(
            handle=img_cfg.handle or self.settings.download.kaggle_handle,
            dataset_root=self.settings.dataset_root,
            catalog_name=img_cfg.catalog_name or self.settings.catalog_name,
            downloader=self.downloader,
            scanner=self.scanner,
            validator=self.validator,
            csv_loader=self.csv_loader,
            image_loader=self.image_loader,
            metadata_generator=self.metadata_generator,
            metadata_store=self.metadata_store,
            cache=self.cache,
            force_download=img_cfg.force_download,
            materialize=img_cfg.materialize,
            link_method=img_cfg.link_method,
            verify_integrity=img_cfg.verify_integrity,
        )

        # Ensure the default catalog is registered so the registry is never empty.
        if self.registry.get_by_name(self.settings.catalog_name) is None:
            self.registry.register(
                name=self.settings.catalog_name,
                source="kaggle",
                root_path=self.settings.catalog_root,
                status=DatasetStatus.PENDING,
            )

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "DatasetManager":
        """Build a fully-wired manager from a YAML config file / environment.

        Args:
            config_path: Optional YAML configuration file. When omitted the
                ``DM_CONFIG_FILE`` environment variable (or defaults) apply.
        """
        settings = load_settings(config_path)
        return cls(settings)

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #

    def download(
        self,
        handle: str | None = None,
        *,
        force: bool = False,
        materialize: bool | None = None,
    ) -> Path:
        """Download (or reuse) the primary Kaggle dataset and materialise it.

        Args:
            handle: Kaggle handle; defaults to the configured one.
            force: Re-download even when a copy is cached.
            materialize: Mirror the download into the managed raw root.
                Defaults to the download configuration value.

        Returns:
            The materialised root path (canonical location under ``raw/``).
        """
        handle = handle or self.settings.download.kaggle_handle
        self._set_status(DatasetStatus.DOWNLOADING)
        self.image_provider.handle = handle
        try:
            path = self.image_provider.ensure(force=force, materialize=materialize)
        except Exception:
            self._set_status(DatasetStatus.FAILED)
            raise
        status = (
            DatasetStatus.DOWNLOADED
            if self.image_provider.status.value != "error"
            else DatasetStatus.FAILED
        )
        self._set_status(status)
        return path

    def scan(self, *, use_cache: bool | None = None, refresh: bool = False) -> DatasetInventory:
        """Scan the managed dataset root and return an inventory.

        Args:
            use_cache: Override the scan cache setting.
            refresh: Force a re-scan, invalidating the cache first.
        """
        root = self._assert_root()
        if refresh:
            self.cache_invalidate(f"scan:{root}")
        return self.scanner.scan(root, use_cache=use_cache is None or use_cache)

    def validate(self, *, report_dir: str | Path | None = None) -> ValidationReport:
        """Validate the scanned inventory and return a detailed report.

        Args:
            report_dir: Optional directory to persist ``validation_report.json``.
        """
        self._set_status(DatasetStatus.VALIDATING)
        inventory = self.scan(use_cache=True)
        report = self.validator.validate(inventory.root, inventory)
        self._set_status(DatasetStatus.VALIDATED if report.passed else DatasetStatus.FAILED)

        out_dir = Path(report_dir) if report_dir else self.settings.state_root
        out_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(out_dir / "validation_report.json")
        return report

    def generate_metadata(self, *, force: bool = False) -> int:
        """Generate (and persist) metadata records for every scanned file.

        Args:
            force: Regenerate records even when the file size is unchanged.

        Returns:
            The number of records written.
        """
        inventory = self.scan(use_cache=True)
        records = self.metadata_generator.generate(
            inventory.root, inventory, force=force
        )
        return len(records)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def inventory(self) -> DatasetInventory:
        """Return the current inventory (served from cache when fresh)."""
        return self.scan(use_cache=True)

    def summary(self) -> DatasetSummary:
        """Return a human + machine readable summary of the dataset."""
        inventory = self.scan(use_cache=True)
        counts = inventory.counts()
        by_year = inventory.by_year()
        by_resolution = inventory.by_resolution()

        index_types = sorted({e.index_type.value for e in inventory.entries})
        resolutions = sorted({e.resolution.value for e in inventory.entries})
        years = sorted(by_year.keys())

        csv_row_estimate: int | None = None
        csv_entries = inventory.csv_files()
        if csv_entries:
            try:
                csv_row_estimate = self.csv_loader.row_count(csv_entries[0].path)
            except Exception:  # noqa: BLE001 - best effort
                csv_row_estimate = None

        return DatasetSummary(
            root=inventory.root,
            total_files=len(inventory.entries),
            total_size_bytes=inventory.total_size(),
            csv_count=counts["csv"],
            geotiff_count=counts["geotiff"],
            other_count=counts["total"] - counts["csv"] - counts["geotiff"],
            ndvi_count=counts["ndvi"],
            evi_count=counts["evi"],
            files_by_year={y: len(by_year[y]) for y in years},
            files_by_resolution={r.value: len(by_resolution[r]) for r in by_resolution},
            years_covered=years,
            index_types_present=index_types,
            resolutions_present=resolutions,
            csv_row_estimate=csv_row_estimate,
        )

    # ------------------------------------------------------------------ #
    # Data access (the only read paths)
    # ------------------------------------------------------------------ #

    def list_csvs(self) -> list[Path]:
        """List every CSV file in the managed dataset root.

        Uses the scanned inventory so internal state directories are
        consistently excluded.
        """
        inventory = self.scan(use_cache=True)
        return [e.path for e in inventory.csv_files()]

    def list_images(
        self,
        *,
        index_type: IndexType | str | None = None,
        resolution: Resolution | str | None = None,
        year: int | None = None,
    ) -> list[Path]:
        """List GeoTIFF files, optionally filtered by index / resolution / year.

        Args:
            index_type: ``IndexType.NDVI``, ``IndexType.EVI`` (or ``"NDVI"``).
            resolution: ``Resolution.R10M``, ``Resolution.R20M`` (or ``"R10m"``).
            year: Exact 4-digit year.
        """
        inventory = self.scan(use_cache=True)
        files = [e.path for e in inventory.geotiff_files()]
        if index_type is not None:
            wanted = index_type.value if isinstance(index_type, IndexType) else str(index_type).upper()
            files = [
                p for p in files
                if _classify_image(p)["index_type"].value == wanted
            ]
        if resolution is not None:
            wanted = resolution.value if isinstance(resolution, Resolution) else str(resolution)
            files = [
                p for p in files
                if _classify_image(p)["resolution"].value == wanted
            ]
        if year is not None:
            files = [p for p in files if _classify_image(p)["year"] == year]
        return files

    def load_csv(
        self,
        path: str | Path,
        *,
        chunksize: int | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load a CSV file (DataFrame, or iterator when ``chunksize`` set).

        Only files under the managed dataset root are accepted, keeping the
        Dataset Manager the sole data access point.
        """
        resolved = self._resolve_within_root(path)
        return self.csv_loader.load(
            resolved, chunksize=chunksize, columns=columns, **kwargs
        )

    def preview_csv(self, path: str | Path, n_rows: int = 5) -> Any:
        """Preview the first rows of a CSV file."""
        resolved = self._resolve_within_root(path)
        return self.csv_loader.preview(resolved, n_rows=n_rows)

    def load_image(
        self,
        path: str | Path,
        *,
        window: tuple[int, int, int, int] | None = None,
        band: int = 1,
    ) -> Any:
        """Read a raster (full band or a bounded window).

        Prefer ``window`` for memory efficiency.
        """
        resolved = self._resolve_within_root(path)
        if window is not None:
            return self.image_loader.read_window(resolved, window=window, band=band)
        return self.image_loader.load(resolved, band=band)

    def image_metadata(self, path: str | Path) -> dict[str, Any]:
        """Lazy header metadata for a raster file."""
        resolved = self._resolve_within_root(path)
        return self.image_loader.read_metadata(resolved).to_dict()

    def load_geometries(
        self,
        source: str | Path,
        *,
        layer: str | None = None,
    ) -> Any:
        """Load administrative boundary geometries (Shapefile / GeoJSON).

        This is the **only** sanctioned way for other modules (STAM, GIS) to
        read boundary data — nobody scans boundary files directly. The source
        must live under the managed dataset root or under ``settings.admin_dir``.

        Args:
            source: Path to a shapefile/GeoJSON (absolute, or relative to the
                managed root or the admin directory).
            layer: Optional layer name for multi-layer files (Fiona).

        Returns:
            A :class:`geopandas.GeoDataFrame` with the boundary geometries.

        Raises:
            DatasetNotFoundError: When the source is missing or disallowed, or
                geopandas is not installed.
        """
        candidate = self._resolve_admin_path(source)
        try:
            import geopandas as gpd  # lazy: only needed for boundaries
        except ImportError as exc:  # pragma: no cover - environment dependency
            raise DatasetNotFoundError(
                "geopandas is required to load boundary geometries; "
                "install with `pip install geopandas`",
                detail=str(candidate),
            ) from exc
        try:
            gdf = gpd.read_file(candidate, layer=layer)
        except Exception as exc:  # noqa: BLE001
            raise DatasetNotFoundError(
                f"Could not read geometry source: {exc}", detail=str(candidate)
            ) from exc
        if gdf is None or len(gdf) == 0:
            raise DatasetNotFoundError(
                "Geometry source contains no features", detail=str(candidate)
            )
        if "geometry" not in gdf.columns:
            raise DatasetNotFoundError(
                "Geometry source has no geometry column", detail=str(candidate)
            )
        if gdf.crs is None:
            logger.warning(
                "Boundary file has no CRS (assumed EPSG:4326)",
                extra={"path": str(candidate)},
            )
        return gdf

    def _resolve_admin_path(self, source: str | Path) -> Path:
        """Resolve + authorise a geometry source path.

        Accepts absolute paths and paths relative to the dataset root or the
        configured admin directory. Raises ``DatasetNotFoundError`` otherwise.
        """
        raw = Path(source).expanduser()
        roots = [self.settings.dataset_root.resolve()]
        if self.settings.admin_dir is not None:
            roots.append(Path(self.settings.admin_dir).resolve())

        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(raw.resolve())  # relative to cwd
            for root in roots:
                candidates.append((root / raw).resolve())

        for candidate in candidates:
            for base in roots:
                try:
                    candidate.relative_to(base)
                except ValueError:
                    continue
                if candidate.exists():
                    return candidate
        raise DatasetNotFoundError(
            f"Geometry source is missing or outside allowed roots: {source}",
            detail={"allowed_roots": [str(r) for r in roots]},
        )

    # ------------------------------------------------------------------ #
    # Provider access (tabular + image data sources)
    # ------------------------------------------------------------------ #

    # -- Tabular (Git-versioned CSVs) ---------------------------------------- #

    def tabular_catalog(self, *, refresh: bool = False) -> TabularCatalog:
        """Discover the Git-versioned tabular datasets (auto-discovery)."""
        return self.tabular_provider.discover(refresh=refresh)

    def tabular_names(self) -> list[str]:
        """Discovered tabular dataset names (sorted, no hardcoded filenames)."""
        return self.tabular_provider.names()

    def load_tabular(
        self,
        name: str,
        *,
        chunksize: int | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load a discovered tabular dataset as a DataFrame (or iterator)."""
        return self.tabular_provider.load(
            name, chunksize=chunksize, columns=columns, **kwargs
        )

    def stream_tabular(self, name: str, chunksize: int, **kwargs: Any) -> Any:
        """Stream a tabular dataset in bounded chunks (memory bounded)."""
        return self.tabular_provider.stream(name, chunksize, **kwargs)

    def tabular_schema(self, name: str) -> dict[str, Any]:
        """Schema / dtype / missing-value profile of a tabular dataset."""
        return self.tabular_provider.schema(name)

    def validate_tabular_schema(self, name: str) -> dict[str, Any]:
        """Run schema validation; returns ``{valid, issues}``."""
        return self.tabular_provider.validate_schema(name)

    def tabular_statistics(self, name: str) -> dict[str, Any]:
        """Numeric column statistics of a tabular dataset."""
        return self.tabular_provider.statistics(name)

    def tabular_missing(self, name: str) -> dict[str, int]:
        """``{column: missing_count}`` for a tabular dataset."""
        return self.tabular_provider.missing_values(name)

    def handle_missing_tabular(
        self,
        name: str,
        strategy: str = "drop",
        *,
        fill_value: Any = None,
        fill_method: str = "mean",
    ) -> Any:
        """Return a copy of the dataset with missing values handled."""
        return self.tabular_provider.handle_missing(
            name, strategy, fill_value=fill_value, fill_method=fill_method
        )

    def join_tabular(
        self, joins: list[TabularJoinSpec], *, how: str | None = None
    ) -> Any:
        """Sequentially join discovered tabular datasets into one frame."""
        return self.tabular_provider.join(joins, how=how)

    def tabular_metadata(self, name: str) -> dict[str, Any]:
        """Tabular dataset metadata (path, size, schema, statistics)."""
        return self.tabular_provider.metadata(name)

    # -- Image (Kaggle Sentinel NDVI / EVI) ------------------------------------ #

    def ensure_image(
        self, *, force: bool = False, materialize: bool | None = None
    ) -> Path:
        """Download (or reuse) the imagery dataset via the image provider."""
        return self.image_provider.ensure(force=force, materialize=materialize)

    def image_location(self) -> ImageDatasetLocation:
        """Current on-disk location / materialisation state of the imagery."""
        return self.image_provider.location()

    def image_catalog(self, *, refresh: bool = False) -> ImageCatalog:
        """Classified imagery catalog (NDVI / EVI, year, resolution)."""
        return self.image_provider.catalog(refresh=refresh)

    def validate_image(
        self, *, report_dir: str | Path | None = None
    ) -> ValidationReport:
        """Validate the imagery dataset through the image provider."""
        return self.image_provider.validate(report_dir=report_dir)

    def generate_image_metadata(self, *, force: bool = False) -> int:
        """Generate metadata records for the imagery dataset."""
        return self.image_provider.generate_metadata(force=force)

    def discover_ndvi(self) -> list[Any]:
        """Discover NDVI rasters (lazy — no pixel data loaded)."""
        return self.image_provider.discover_ndvi()

    def discover_evi(self) -> list[Any]:
        """Discover EVI rasters (lazy — no pixel data loaded)."""
        return self.image_provider.discover_evi()

    def read_image(
        self,
        path: str | Path,
        *,
        window: tuple[int, int, int, int] | None = None,
        band: int = 1,
    ) -> Any:
        """Read a raster band (or bounded window) through the provider."""
        return self.image_provider.read(path, window=window, band=band)

    def patch_image(self, request: PatchRequest) -> Any:
        """Retrieve a square raster patch around a geographic center point."""
        return self.image_provider.patch(request)

    def image_historical_context(
        self,
        *,
        window_months: Sequence[int] | set[int] | None = None,
        index_type: str | None = None,
        resolution: str | None = None,
        years: list[int] | None = None,
    ) -> HistoricalContext:
        """Temporal imagery availability via the image provider."""
        months: list[int] | None = None
        if window_months is not None:
            months = [int(m) for m in window_months]
        return self.image_provider.get_historical_context(
            window_months=months,
            index_type=index_type,
            resolution=resolution,
            years=years,
        )

    def provider_manifests(self) -> dict[str, Any]:
        """Introspection of every configured provider (for diagnostics)."""
        return {
            self.tabular_provider.name: self.tabular_provider.manifest().to_dict(),
            self.image_provider.name: self.image_provider.manifest().to_dict(),
        }

    # ------------------------------------------------------------------ #
    # Metadata access
    # ------------------------------------------------------------------ #

    def get_metadata(self, path: str | Path) -> MetadataRecord | None:
        """Return the metadata record for a file, if generated."""
        resolved = self._resolve_within_root(path, must_exist=False)
        return self.metadata_store.get(resolved)

    def query_metadata(self, **filters: Any) -> list[MetadataRecord]:
        """Query metadata records (e.g. ``year=2022, index_type="NDVI"``)."""
        return self.metadata_store.query(**filters)

    def metadata_count(self) -> int:
        return self.metadata_store.count()

    def get_historical_context(
        self,
        *,
        window_months: Sequence[int] | set[int] | None = None,
        index_type: str | None = None,
        resolution: str | None = None,
        years: list[int] | None = None,
    ) -> HistoricalContext:
        """Temporal availability for a recurring season window.

        Aggregates satellite metadata records by year for a *recurring*
        calendar window (e.g. the Kharif months Jun-Oct in every year). This
        is the "same season across all years" evidence STAM attaches to an
        observation before model inference, so a farmer only needs a location.

        Args:
            window_months: Calendar months the season occupies (crossing
                seasons like Rabi Nov-Mar use ``[11, 12, 1, 2, 3]``). None
                aggregates every record in the catalog.
            index_type: Restrict to ``"NDVI"`` / ``"EVI"`` (None for all).
            resolution: Restrict to ``"R10m"`` / ``"R20m"``.
            years: Restrict the summary to specific years.

        Returns:
            A :class:`HistoricalContext` with per-year record counts.
        """
        records = self.query_metadata(
            category="geotiff", index_type=index_type, resolution=resolution
        )
        months: set[int] | None = None
        if window_months is not None:
            months = {int(m) for m in window_months}
        if years is not None:
            wanted = set(years)
            records = [r for r in records if r.year in wanted]
        if months is not None:
            records = [
                r for r in records
                if r.observation_date is not None and r.observation_date.month in months
            ]

        per_year: dict[int, dict[str, int]] = {}
        for record in records:
            if record.year is None or record.observation_date is None:
                continue
            bucket = per_year.setdefault(record.year, {})
            bucket["records"] = bucket.get("records", 0) + 1
            index_key = record.index_type.value.lower()
            bucket[index_key] = bucket.get(index_key, 0) + 1

        return HistoricalContext(
            window_months=sorted(months) if months is not None else None,
            years=sorted(per_year),
            per_year=per_year,
            total_records=len(records),
            generated_at=datetime.now(),
        )

    def export_metadata_parquet(self, path: str | Path) -> Path:
        """Export the full metadata table to Parquet for analytics."""
        return self.metadata_store.export_parquet(Path(path))

    # ------------------------------------------------------------------ #
    # Cache control
    # ------------------------------------------------------------------ #

    def cache_get(self, key: str) -> Any | None:
        return self.cache.get(key)

    def cache_set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        self.cache.set(key, value, ttl_seconds=ttl_seconds)

    def cache_invalidate(self, key_or_prefix: str) -> int:
        """Invalidate a cache key or prefix (e.g. ``"scan:"``)."""
        if self.cache.get(key_or_prefix) is not None:
            self.cache.delete(key_or_prefix)
            return 1
        return self.cache.delete_prefix(key_or_prefix)

    def cache_clear(self) -> int:
        return self.cache.clear()

    def cache_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.cache.enabled,
            "db_path": str(self.settings.cache_db_path()),
            "entries": self.cache.size() if hasattr(self.cache, "size") else None,
        }

    # ------------------------------------------------------------------ #
    # Registry + versioning
    # ------------------------------------------------------------------ #

    def register(
        self,
        *,
        name: str | None = None,
        source: str = "kaggle",
        root_path: str | Path | None = None,
    ) -> int:
        """Register the dataset in the registry and return its id."""
        name = name or self.settings.catalog_name
        root = Path(root_path) if root_path else self.settings.catalog_root
        existing = self.registry.get_by_name(name)
        if existing is not None:
            return int(existing["dataset_id"])
        return self.registry.register(
            name=name, source=source, root_path=root, status=DatasetStatus.READY
        )

    def registry_entries(self) -> list[dict[str, Any]]:
        return self.registry.list()

    def registry_status(self) -> dict[str, Any]:
        entries = self.registry.list()
        return {
            "count": len(entries),
            "entries": entries,
        }

    def current_version(self) -> str | None:
        entry = self.registry.get_by_name(self.settings.catalog_name)
        if entry is None:
            return None
        return self.version_manager.current(int(entry["dataset_id"]))

    def list_versions(self) -> list[VersionEntry]:
        entry = self.registry.get_by_name(self.settings.catalog_name)
        if entry is None:
            return []
        return self.version_manager.list(int(entry["dataset_id"]))

    def bump_version(self, part: str = "patch", *, message: str = "") -> VersionEntry:
        """Bump the dataset version (major/minor/patch) and snapshot it.

        The snapshot records a fresh tree checksum and file count so each
        version is independently verifiable.
        """
        from .version_manager import bump_version as _bump_version

        dataset_id = self._default_dataset_id()
        current = self.current_version() or "0.0.0"
        next_version = _bump_version(current, part)
        checksum, file_count = self._compute_checksum()
        return self.version_manager.snapshot(
            dataset_id,
            next_version,
            message=message or f"bump {part}",
            checksum=checksum,
            file_count=file_count,
        )

    def rollback_version(self, version: str) -> VersionEntry:
        """Roll the dataset back to a previously snapshotted version."""
        return self.version_manager.rollback(self._default_dataset_id(), version)

    def snapshot_version(
        self, version: str, *, message: str, checksum: str | None, file_count: int
    ) -> VersionEntry:
        return self.version_manager.snapshot(
            self._default_dataset_id(), version, message=message,
            checksum=checksum, file_count=file_count,
        )

    # ------------------------------------------------------------------ #
    # Info & lifecycle
    # ------------------------------------------------------------------ #

    def info(self) -> dict[str, Any]:
        """Environment + configuration summary (for diagnostics / CLI)."""
        return {
            "python": platform.python_version(),
            "platform": sys.platform,
            "dataset_root": str(self.settings.dataset_root.resolve()),
            "catalog_root": str(self.settings.catalog_root),
            "state_root": str(self.settings.state_root),
            "metadata_store": self.settings.metadata.store_type,
            "cache_enabled": self.settings.cache.enabled,
            "kaggle_handle": self.settings.download.kaggle_handle,
            "providers": self.provider_manifests(),
            "dependencies": {
                "kagglehub": _import_version("kagglehub"),
                "rasterio": _import_version("rasterio"),
                "pandas": _import_version("pandas"),
            },
        }

    def close(self) -> None:
        """Release resources (metadata store, caches)."""
        try:
            self.metadata_store.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info("DatasetManager closed")

    def __enter__(self) -> "DatasetManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _assert_root(self) -> Path:
        root = self.settings.dataset_root.resolve()
        if not root.is_dir():
            raise DatasetNotFoundError(
                f"Dataset root does not exist: {root}. Run `download` first.",
                detail=str(root),
            )
        return root

    def _resolve_within_root(self, path: str | Path, *, must_exist: bool = True) -> Path:
        """Resolve a user-supplied path, enforcing it lives in the dataset root."""
        root = self._assert_root()
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise DatasetNotFoundError(
                f"Path is outside the managed dataset root: {candidate}",
                detail=str(root),
            ) from exc
        if must_exist and not candidate.exists():
            raise DatasetNotFoundError(f"File not found: {candidate}", detail=str(candidate))
        return candidate

    def _set_status(self, status: DatasetStatus) -> None:
        entry = self.registry.get_by_name(self.settings.catalog_name)
        if entry is not None:
            self.registry.update_status(int(entry["dataset_id"]), status)

    def _default_dataset_id(self) -> int:
        entry = self.registry.get_by_name(self.settings.catalog_name)
        if entry is None:
            return self.register()
        return int(entry["dataset_id"])

    def _compute_checksum(self) -> tuple[str, int]:
        """Compute a fast tree checksum + file count over the dataset root.

        The checksum is a SHA-256 over ``(root, file_count, total_size,
        max_mtime)`` — a cheap integrity fingerprint that changes when any
        file is added/removed/resized/touched. Full content hashing is
        available separately via metadata records (``sha256``).
        """
        import hashlib
        from .utils import tree_signature

        root = self._assert_root()
        from .utils import DEFAULT_EXCLUDE_DIRS

        count, total, max_mtime = tree_signature(root, exclude_dirs=DEFAULT_EXCLUDE_DIRS)
        digest = hashlib.sha256()
        digest.update(str(root).encode("utf-8"))
        digest.update(f"{count}:{int(total)}:{max_mtime:.3f}".encode("utf-8"))
        return digest.hexdigest(), count

    def _configure_logging(self) -> None:
        log = self.settings.logging
        if log.dir is not None or log.console:
            setup_logging(
                level=log.level,
                log_dir=log.dir,
                max_bytes=log.max_bytes,
                backup_count=log.backup_count,
                json_format=log.json_format,
                console=log.console,
            )


def _classify_image(path: Path) -> dict[str, Any]:
    """Re-classify a raster path for filtering helpers (no metadata store hit)."""
    from .utils import (
        classify_index_type_from_path,
        classify_resolution_from_path,
        extract_year_from_path,
    )

    return {
        "index_type": classify_index_type_from_path(path),
        "resolution": classify_resolution_from_path(path),
        "year": extract_year_from_path(path),
    }


def _import_version(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "present")
    except Exception:  # noqa: BLE001
        return "present"
