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
from typing import Any, Iterable, Sequence

from .cache_manager import CacheManager
from .config import Settings, load_settings
from .csv_loader import PandasCSVLoader
from .dataset_registry import SQLiteRegistry
from .downloader import KaggleDownloader
from .exceptions import DatasetNotFoundError
from .historical_context_builder import HistoricalContextBuilderImpl
from .image_loader import RasterioImageLoader
from .interfaces import (
    Cache,
    CSVLoader,
    Downloader,
    HistoricalContextBuilder,
    ImageLoader,
    MetadataGenerator,
    MetadataStore,
    PatchExtractor,
    ProviderRegistry,
    Registry,
    Scanner,
    SpatialIndex,
    Validator,
    VersionManager,
)
from .logger import get_logger, setup_logging
from .manager_paths import ensure_state_dirs
from .metadata import MetadataGeneratorImpl, SQLiteMetadataStore
from .metadata_repository import MetadataRepository
from .models import (
    DatasetInventory,
    DatasetStatistics,
    DatasetStatus,
    DatasetSummary,
    FileCategory,
    HistoricalContext,
    HistoricalObservationSet,
    IndexType,
    LocationResult,
    MetadataRecord,
    Resolution,
    SpatialMetadata,
    SpatialRecord,
    ValidationReport,
    VersionEntry,
)
from .patch_extractor import PatchExtractorImpl
from .provider_registry import ProviderRegistryImpl
from .providers import GitRepositoryTabularProvider, KaggleHubImageProvider
from .providers.models import (
    ImageCatalog,
    ImageDatasetLocation,
    PatchRequest,
    TabularCatalog,
    TabularJoinSpec,
)
from .reports import generate_reports
from .scanner import DatasetScanner
from .spatial_index import SpatialIndexImpl, build_records_from_frame
from .statistics import compute_statistics
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

        # -- Provider registry (the only way data sources are touched) ----------- #
        # Providers are registered, then resolved through the registry — the
        # manager never constructs or holds them directly.
        tab_cfg = self.settings.providers.tabular
        tabular_provider = GitRepositoryTabularProvider(
            root=self.settings.tabular_root,
            loader=self.csv_loader,
            patterns=list(tab_cfg.patterns),
        )
        img_cfg = self.settings.providers.image
        image_provider = KaggleHubImageProvider(
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

        self.provider_registry: ProviderRegistry = ProviderRegistryImpl()
        self.provider_registry.register(
            "git_repository_tabular",
            "tabular",
            tabular_provider,
            enabled=True,
            priority=100,
            config={"patterns": list(tab_cfg.patterns), "chunk_size": tab_cfg.chunk_size},
        )
        self.provider_registry.register(
            "kaggle_hub_image",
            "image",
            image_provider,
            enabled=True,
            priority=100,
            config={"handle": image_provider.handle},
        )
        self._apply_provider_registry_config()

        # Legacy R1.2 attributes — wired through the registry without the
        # enabled check (a registered-but-disabled provider is still owned by
        # the manager; the registry enforces availability for consumers).
        self.tabular_provider = self._registered_provider("git_repository_tabular")
        self.image_provider = self._registered_provider("kaggle_hub_image")

        # -- R2.2 extended metadata repository (same metadata.db) ----------------- #
        self.metadata_repository = MetadataRepository(self.settings.metadata_db_path())

        # -- R2.2 spatial index (auto-built from tabular location data) ----------- #
        self.spatial_index: SpatialIndex = SpatialIndexImpl()
        self._auto_build_spatial_index()

        # Wire the validator's extended checks (spatial / provider / metadata).
        self._wire_validator_extensions()

        # -- R2.2 patch extractor + historical context builder -------------------- #
        self.patch_extractor: PatchExtractor = PatchExtractorImpl(
            self.image_provider,
            metadata_repository=self.metadata_repository,
        )
        self.historical_context_builder: HistoricalContextBuilder = (
            HistoricalContextBuilderImpl(
                tabular_provider=self.tabular_provider,
                image_provider=self.image_provider,
                metadata_store=self.metadata_store,
                spatial_index=self.spatial_index,
                metadata_repository=self.metadata_repository,
            )
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
    # Provider registry configuration
    # ------------------------------------------------------------------ #

    def _apply_provider_registry_config(self) -> None:
        """Apply ``providers.registry`` settings (enable/disable/priority).

        Matching names override the default registration; unknown entries of a
        supported kind are registered as additional providers (multi-provider
        setups / future plugins).
        """
        for entry in self.settings.providers.registry.providers:
            name = entry.name
            if self.provider_registry.has(name):
                existing = self.provider_registry.resolve(name)
                self.provider_registry.register(
                    name,
                    entry.kind,
                    existing,
                    enabled=entry.enabled,
                    priority=entry.priority,
                    config=entry.config,
                )
                continue
            if entry.kind == "tabular":
                root = entry.config.get("root") or self.settings.tabular_root
                provider = GitRepositoryTabularProvider(
                    root=root,
                    name=name,
                    loader=self.csv_loader,
                    patterns=entry.config.get("patterns") or ["*.csv"],
                )
                self.provider_registry.register(
                    name, "tabular", provider,
                    enabled=entry.enabled, priority=entry.priority, config=entry.config,
                )
            elif entry.kind == "image":
                provider = KaggleHubImageProvider(
                    handle=entry.config.get("handle") or self.settings.download.kaggle_handle,
                    name=name,
                    dataset_root=entry.config.get("dataset_root") or self.settings.dataset_root,
                    catalog_name=entry.config.get("catalog_name") or self.settings.catalog_name,
                    downloader=self.downloader,
                    scanner=self.scanner,
                    validator=self.validator,
                    csv_loader=self.csv_loader,
                    image_loader=self.image_loader,
                    metadata_generator=self.metadata_generator,
                    metadata_store=self.metadata_store,
                    cache=self.cache,
                )
                self.provider_registry.register(
                    name, "image", provider,
                    enabled=entry.enabled, priority=entry.priority, config=entry.config,
                )
            else:
                logger.warning(
                    "Skipping unsupported provider kind from config",
                    extra={"provider_name": name, "kind": entry.kind},
                )

    def _registered_provider(self, name: str) -> Any:
        """Return the provider instance for ``name`` regardless of enabled state."""
        for registration in self.provider_registry.registrations():
            if registration.name == name:
                return registration.provider
        raise DatasetNotFoundError(
            f"Provider not registered: {name}",
            detail={"registered": self.provider_registry.names()},
        )

    def _auto_build_spatial_index(self) -> None:
        """Best-effort spatial index built from tabular location data.

        Scans discovered tabular datasets for a name column plus latitude /
        longitude columns and indexes every usable row. Datasets without such
        columns are skipped. The result is persisted to the metadata
        repository when available.
        """
        records: list[SpatialRecord] = []
        try:
            names = self.tabular_provider.names()
        except Exception:  # noqa: BLE001 - no tabular data yet
            names = []
        for name in names:
            try:
                columns = self.tabular_provider.schema(name).get("columns", [])
            except Exception:  # noqa: BLE001 - best effort
                continue
            name_col = _find_spatial_name_column(columns)
            lat_col = _find_spatial_lat_column(columns)
            lon_col = _find_spatial_lon_column(columns)
            if not (name_col and lat_col and lon_col):
                continue
            try:
                frame = self.tabular_provider.load(name, chunksize=None)
            except Exception:  # noqa: BLE001 - best effort
                continue
            if len(frame) > 50_000:
                frame = frame.head(50_000)
            kind = "village" if "village" in name_col.lower() else "district"
            records.extend(
                build_records_from_frame(
                    frame,
                    name_col=name_col,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    kind=kind,
                    district_col=(
                        "district"
                        if "district" in {c.lower() for c in columns}
                        else None
                    ),
                )
            )
        if records:
            self.spatial_index.build(records)
            try:
                self.metadata_repository.save_spatial_many(records)
            except Exception:  # noqa: BLE001 - persistence is best-effort
                pass
            logger.info("Auto-built spatial index", extra={"records": len(records)})

    def _wire_validator_extensions(self) -> None:
        """Expose the R2.2 stores to the validator for extended checks.

        The validator is constructed before the registry / spatial index exist,
        so the extended attributes are attached here. Injected test fakes are
        left untouched when they do not accept attributes.
        """
        for attr in ("spatial_index", "provider_registry", "metadata_repository"):
            try:
                setattr(self.validator, attr, getattr(self, attr))
            except Exception:  # noqa: BLE001 - injected fakes may be read-only
                pass
        # Persist provider metadata into the extended repository (best-effort).
        try:
            for registration in self.provider_registry.registrations():
                self.metadata_repository.save_provider(registration)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

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
        resolved = self._resolve_within_root(
            path, extra_roots=self._image_source_roots()
        )
        if window is not None:
            return self.image_loader.read_window(resolved, window=window, band=band)
        return self.image_loader.load(resolved, band=band)

    def image_metadata(self, path: str | Path) -> dict[str, Any]:
        """Lazy header metadata for a raster file."""
        resolved = self._resolve_within_root(
            path, extra_roots=self._image_source_roots()
        )
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
        """Introspection of every registered provider (for diagnostics)."""
        manifests: dict[str, Any] = {}
        for name in self.provider_registry.names():
            try:
                provider = self.provider_registry.resolve(name)
                manifests[name] = provider.manifest().to_dict()
            except Exception as exc:  # noqa: BLE001 - per-provider best-effort
                manifests[name] = {"error": str(exc)}
        return manifests

    # ------------------------------------------------------------------ #
    # R2.2 — Multi-source API
    # ------------------------------------------------------------------ #

    def load_tabular(self, name: str) -> Any:
        """Load a full tabular dataset as a DataFrame (via its provider)."""
        return self.tabular_provider.load(name, chunksize=None)

    def get_csv(self, name: str) -> Any:
        """Alias of :meth:`load_tabular` (R1.2-compatible naming)."""
        return self.load_tabular(name)

    def load_images(
        self,
        *,
        index_type: str | None = None,
        resolution: str | None = None,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load raster arrays matching the given index / resolution / year.

        Returns one dict per raster: ``{path, index_type, resolution, year,
        array}``. No preprocessing is applied.
        """
        matches = self._matching_entries(
            index_type=index_type, resolution=resolution, year=year
        )
        images: list[dict[str, Any]] = []
        for entry in matches:
            images.append(
                {
                    "path": str(entry.path),
                    "index_type": entry.index_type.value,
                    "resolution": entry.resolution.value,
                    "year": entry.year,
                    "array": self.image_provider.read(str(entry.path)),
                }
            )
        return images

    def get_image(
        self,
        index_type: str,
        *,
        resolution: str | None = None,
        year: int | None = None,
    ) -> Any:
        """Read a single best-matching raster as a NumPy array."""
        matches = self._matching_entries(
            index_type=index_type, resolution=resolution, year=year
        )
        if not matches:
            raise DatasetNotFoundError(
                "No matching raster for the requested image",
                detail={
                    "index_type": index_type,
                    "resolution": resolution,
                    "year": year,
                },
            )
        return self.image_provider.read(str(matches[0].path))

    def get_patch(
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
        """Extract a ``size`` x ``size`` geographic patch (raw NumPy array).

        The best raster for the index type / resolution / year is located and
        the point is converted into the raster's CRS automatically.
        """
        return self.patch_extractor.extract(
            latitude,
            longitude,
            size,
            index_type=index_type,
            resolution=resolution,
            year=year,
            band=band,
            padding=padding,
        )

    def get_location(
        self,
        *,
        name: str | None = None,
        kind: str = "village",
        latitude: float | None = None,
        longitude: float | None = None,
        k: int = 1,
        radius_km: float | None = None,
        tolerance: float = 0.01,
    ) -> LocationResult:
        """Resolve a named or coordinate-based location.

        ``name`` is looked up by kind (``village`` / ``district``);
        coordinates use nearest-neighbour / radius / tolerance matching.
        """
        if name is not None:
            if kind == "village":
                records = self.spatial_index.lookup_village(name)
            elif kind == "district":
                records = self.spatial_index.lookup_district(name)
            else:
                records = [
                    r for r in self.spatial_index.records()
                    if r.kind == kind and r.name == name
                ]
            return LocationResult(
                found=bool(records),
                records=records,
                query={"name": name, "kind": kind},
            )
        if latitude is None or longitude is None:
            raise ValueError("get_location requires name or latitude/longitude")
        if radius_km is not None:
            records = self.spatial_index.within_radius(latitude, longitude, radius_km)
        else:
            records = self.spatial_index.search_coordinates(
                latitude, longitude, tolerance=tolerance
            )
        if not records:
            nearest = self.spatial_index.nearest(latitude, longitude, k=k)
            records = [r for r, _dist in nearest]
        return LocationResult(
            found=bool(records),
            records=records,
            query={"latitude": latitude, "longitude": longitude, "k": k},
        )

    def get_available_years(
        self,
        *,
        index_type: str | None = None,
        resolution: str | None = None,
    ) -> list[int]:
        """Years present in the imagery catalog (optionally filtered)."""
        return sorted(
            {
                e.year
                for e in self._matching_entries(
                    index_type=index_type, resolution=resolution
                )
                if e.year is not None
            }
        )

    def get_available_indices(self, *, resolution: str | None = None) -> list[str]:
        """Index types (e.g. ``NDVI``, ``EVI``) present in the catalog."""
        return sorted(
            {
                e.index_type.value
                for e in self._matching_entries(resolution=resolution)
                if e.index_type is not None
            }
        )

    def get_resolutions(self, *, index_type: str | None = None) -> list[str]:
        """Resolutions present in the catalog (optionally filtered)."""
        return sorted(
            {
                e.resolution.value
                for e in self._matching_entries(index_type=index_type)
                if e.resolution is not None
            }
        )

    def statistics(self) -> DatasetStatistics:
        """Aggregate dataset statistics across tabular and image sources."""
        return compute_statistics(
            tabular_provider=self.tabular_provider, image_provider=self.image_provider
        )

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search tabular datasets, imagery and providers for ``query``."""
        needle = query.lower()
        hits: list[dict[str, Any]] = []
        for name in self.tabular_provider.names():
            if needle in name.lower():
                hits.append({"type": "tabular", "name": name})
        for entry in self._catalog_entries():
            haystack = f"{entry.path} {entry.index_type.value} {entry.resolution.value}".lower()
            if needle in haystack:
                hits.append(
                    {
                        "type": "image",
                        "path": str(entry.path),
                        "index_type": entry.index_type.value,
                        "resolution": entry.resolution.value,
                        "year": entry.year,
                    }
                )
        for name in self.provider_registry.names():
            if needle in name.lower():
                hits.append({"type": "provider", "name": name})
        return hits[:limit]

    def availability(self) -> dict[str, bool]:
        """``{provider: available}`` for every registered provider."""
        return self.provider_registry.availability()

    def health(self) -> dict[str, Any]:
        """Health snapshot of every registered provider."""
        return self.provider_registry.health()

    def discovery(self) -> list[dict[str, Any]]:
        """Plain registration records for every provider."""
        return self.provider_registry.discovery()

    def spatial_metadata(self) -> SpatialMetadata:
        """Aggregate statistics of the spatial index."""
        return self.spatial_index.metadata()

    def temporal_metadata(
        self,
        *,
        index_type: str | None = None,
        year: int | None = None,
        resolution: str | None = None,
    ) -> list[dict[str, Any]]:
        """Per index/year/resolution temporal availability records."""
        return self.metadata_repository.list_temporal(
            index_type=index_type, year=year, resolution=resolution
        )

    def provider_metadata(self) -> list[dict[str, Any]]:
        """Persisted provider metadata from the extended repository."""
        return self.metadata_repository.list_providers()

    def list_patches(self, limit: int = 100) -> list[dict[str, Any]]:
        """Recent extracted patch records (from the metadata repository)."""
        return self.metadata_repository.list_patches(limit=limit)

    def reports(self, report_dir: str | Path | None = None) -> list[Path]:
        """Generate the R2.2 reports; returns the written report paths."""
        return generate_reports(self, report_dir=report_dir)

    def build_historical_context(
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
        """Multi-year, per-location observation context (no STAM inference)."""
        return self.historical_context_builder.build(
            village=village,
            district=district,
            latitude=latitude,
            longitude=longitude,
            index_type=index_type,
            resolution=resolution,
            years=years,
        )

    # -- R2.2 helpers ----------------------------------------------------------- #

    def _catalog_entries(self) -> list[Any]:
        try:
            return list(self.image_provider.catalog().entries)
        except Exception as exc:  # noqa: BLE001 - imagery may not be materialised
            logger.debug("Imagery catalog unavailable", extra={"error": str(exc)})
            return []

    def _matching_entries(
        self,
        *,
        index_type: str | None = None,
        resolution: str | None = None,
        year: int | None = None,
    ) -> list[Any]:
        entries = [e for e in self._catalog_entries() if e.category is FileCategory.GEOTIFF]
        if index_type is not None:
            needle = index_type.upper()
            entries = [e for e in entries if e.index_type is not None and e.index_type.value == needle]
        if resolution is not None:
            needle = resolution.upper()
            entries = [
                e for e in entries
                if e.resolution is not None and e.resolution.value.upper() == needle
            ]
        if year is not None:
            entries = [e for e in entries if e.year == year]
        return entries

    # ------------------------------------------------------------------ #
    # Metadata access
    # ------------------------------------------------------------------ #

    def get_metadata(self, path: str | Path) -> MetadataRecord | None:
        """Return the metadata record for a file, if generated."""
        resolved = self._resolve_within_root(
            path, must_exist=False, extra_roots=self._image_source_roots()
        )
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

    def _image_source_roots(self) -> list[Path]:
        """Filesystem roots the image provider may read from (e.g. the
        Kaggle ``/kaggle/input`` mount when imagery is attached)."""
        try:
            provider = self._registered_provider("kaggle_hub_image")
            roots = getattr(provider, "source_roots", None)
            if callable(roots):
                return [Path(r) for r in roots()]
        except Exception:  # noqa: BLE001 - provider not registered
            pass
        return []

    def _resolve_within_root(
        self,
        path: str | Path,
        *,
        must_exist: bool = True,
        extra_roots: Iterable[str | Path] | None = None,
    ) -> Path:
        """Resolve a user-supplied path, enforcing it lives in the dataset root
        (or in one of ``extra_roots`` — e.g. the image provider's mount)."""
        root = self._assert_root()
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        allowed = [root]
        if extra_roots:
            for entry in extra_roots:
                resolved = Path(entry).expanduser().resolve()
                if resolved not in allowed:
                    allowed.append(resolved)
        if not any(self._is_within(candidate, base) for base in allowed):
            raise DatasetNotFoundError(
                f"Path is outside the managed dataset root: {candidate}",
                detail=str(root),
            )
        if must_exist and not candidate.exists():
            raise DatasetNotFoundError(f"File not found: {candidate}", detail=str(candidate))
        return candidate

    @staticmethod
    def _is_within(candidate: Path, base: Path) -> bool:
        try:
            candidate.relative_to(base)
        except ValueError:
            return False
        return True

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


def _find_spatial_name_column(columns: list[str]) -> str | None:
    """Best candidate for a location-name column (village/district/...)."""
    lowered = [c.lower().strip() for c in columns]
    for needle in ("village", "district", "taluk", "tehsil", "block", "mandal", "state"):
        if needle in lowered:
            return columns[lowered.index(needle)]
    for i, col in enumerate(lowered):
        if col in ("name", "place", "location") or col.endswith("_name"):
            return columns[i]
    return None


def _find_spatial_lat_column(columns: list[str]) -> str | None:
    lowered = [c.lower().strip() for c in columns]
    for i, col in enumerate(lowered):
        if col in ("lat", "latitude", "lat_deg", "latitude_n"):
            return columns[i]
    for i, col in enumerate(lowered):
        if col.endswith("latitude") or col.startswith("lat") or "latitude" in col:
            return columns[i]
    return None


def _find_spatial_lon_column(columns: list[str]) -> str | None:
    lowered = [c.lower().strip() for c in columns]
    for i, col in enumerate(lowered):
        if col in ("lon", "lng", "longitude", "lon_deg", "longitude_e"):
            return columns[i]
    for i, col in enumerate(lowered):
        if col.endswith("longitude") or col.startswith("lon") or "longitude" in col:
            return columns[i]
    return None


def _import_version(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "present")
    except Exception:  # noqa: BLE001
        return "present"
