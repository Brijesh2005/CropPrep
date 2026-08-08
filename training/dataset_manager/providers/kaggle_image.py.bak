"""Kaggle Hub imagery data provider.

:class:`KaggleHubImageProvider` serves the **large Sentinel-2 imagery**
dataset (NDVI / EVI GeoTIFFs for 2018-2025) hosted on Kaggle:

    shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada

These images must NEVER be committed to GitHub — they are acquired through
``kagglehub`` at runtime (download or reuse of the local Kaggle cache).

Responsibilities (per the R1.2 specification):

* **Download or reuse** the Kaggle dataset and return its location.
* **Validate imagery** — structural / integrity validation via the scanner +
  validator.
* **Discover NDVI / EVI** rasters (classified, lazy — no pixel data loaded).
* **Lazy GeoTIFF access** — header-only metadata and windowed reads.
* **Metadata generation** — per-file metadata records.
* **Patch retrieval interface** — square patches around a geographic center
  point. **No raster preprocessing happens at this layer.**

The provider reuses the existing Dataset Manager components (downloader,
scanner, validator, image loader, metadata generator) as its engines — it
orchestrates them behind the provider contract.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_KAGGLE_HANDLE
from ..csv_loader import PandasCSVLoader
from ..downloader import KaggleDownloader
from ..exceptions import DatasetNotFoundError
from ..image_loader import RasterioImageLoader
from ..interfaces import (
    Cache,
    Downloader,
    ImageLoader,
    MetadataGenerator,
    MetadataStore,
    Scanner,
    Validator,
)
from ..logger import get_logger
from ..metadata import MetadataGeneratorImpl, SQLiteMetadataStore
from ..models import (
    DatasetInventory,
    DatasetStatus,
    FileCategory,
    FileEntry,
    HistoricalContext,
    IndexType,
    MetadataRecord,
    RasterMetadata,
    Resolution,
    ValidationReport,
)
from ..scanner import DatasetScanner
from ..utils import (
    classify_index_type_from_path,
    classify_resolution_from_path,
    extract_year_from_path,
    parse_observation_date,
)
from ..validator import DatasetValidator
from .base import ImageProvider
from .models import (
    ImageCatalog,
    ImageDatasetLocation,
    PatchRequest,
    ProviderCapabilities,
    ProviderManifest,
    ProviderStatus,
)

logger = get_logger("image_provider")

_DEFAULT_VERSION = "latest"

#: Raster characteristics validated by the image provider.
_EXPECTED_BANDS = 1


class KaggleHubImageProvider(ImageProvider):
    """Concrete :class:`ImageProvider` backed by the Kaggle Sentinel dataset.

    Args:
        handle: Kaggle dataset handle (default: the CropFusion image dataset).
        dataset_root: Root of the managed dataset tree (``raw/`` and
            ``.cropfusion/`` live below it).
        catalog_name: Catalog directory name inside ``dataset_root/raw``.
        downloader: Optional :class:`Downloader` (default: Kaggle).
        scanner: Optional :class:`Scanner`.
        validator: Optional :class:`Validator`.
        image_loader: Optional :class:`ImageLoader`.
        metadata_generator: Optional :class:`MetadataGenerator`.
        metadata_store: Optional :class:`MetadataStore`.
        cache: Optional :class:`Cache`.
        force_download: Re-download even when a cached copy exists.
        materialize: Mirror downloads into ``dataset_root/raw/catalog_name``.
        link_method: ``hardlink`` | ``copy`` for materialisation.
        verify_integrity: Post-download integrity pre-flight.
    """

    name = "kaggle_hub_image"
    kind = "image"

    def __init__(
        self,
        handle: str | None = None,
        *,
        name: str | None = None,
        dataset_root: str | Path | None = None,
        catalog_name: str = "kaggle-crop-yield",
        downloader: Downloader | None = None,
        scanner: Scanner | None = None,
        validator: Validator | None = None,
        image_loader: ImageLoader | None = None,
        csv_loader: CSVLoader | None = None,
        metadata_generator: MetadataGenerator | None = None,
        metadata_store: MetadataStore | None = None,
        cache: Cache | None = None,
        force_download: bool = False,
        materialize: bool = True,
        link_method: str = "hardlink",
        verify_integrity: bool = True,
    ) -> None:
        self.name = name or self.name
        self.handle = handle or DEFAULT_KAGGLE_HANDLE
        self.dataset_root = Path(dataset_root) if dataset_root else Path("datasets")
        self.catalog_name = catalog_name
        self.force_download = force_download
        self.materialize = materialize
        self.link_method = link_method
        self.verify_integrity = verify_integrity

        self.downloader = downloader or KaggleDownloader()
        self.scanner = scanner or DatasetScanner()
        self.image_loader = image_loader or RasterioImageLoader()
        self.csv_loader = csv_loader if csv_loader is not None else PandasCSVLoader()
        self.metadata_store = metadata_store
        if self.metadata_store is None:
            self.metadata_store = SQLiteMetadataStore(
                self.state_root / "metadata.db"
            )
        self.validator = validator or DatasetValidator(
            image_loader=self.image_loader,
            metadata_store=self.metadata_store,
        )
        self.metadata_generator = metadata_generator or MetadataGeneratorImpl(
            csv_loader=self.csv_loader,
            image_loader=self.image_loader,
            store=self.metadata_store,
        )
        self.cache = cache

        self._status = ProviderStatus.NOT_INITIALIZED
        self._catalog: ImageCatalog | None = None
        self._inventory: DatasetInventory | None = None
        self._mounted_root: Path | None = None

    # ------------------------------------------------------------------ #
    # Derived paths
    # ------------------------------------------------------------------ #

    def _kaggle_input_root(self) -> Path | None:
        """Root of the imagery dataset mounted under ``/kaggle/input`` (if any).

        Kaggle attaches datasets read-only; the ~148 GB Sentinel imagery is
        consumed **in place** through the mount and never re-downloaded or
        duplicated via ``kagglehub`` on the training box.
        """
        if self._mounted_root is None:
            slug = self.handle.split("/")[-1]
            candidate = Path("/kaggle/input") / slug
            self._mounted_root = candidate if candidate.is_dir() else None
        return self._mounted_root

    @property
    def raw_root(self) -> Path:
        return self.dataset_root / "raw"

    @property
    def catalog_root(self) -> Path:
        """Canonical location of the materialised imagery dataset."""
        return self.raw_root / self.catalog_name

    @property
    def state_root(self) -> Path:
        return self.dataset_root / ".cropfusion"

    def source_root(self) -> Path:
        """Effective imagery root: the Kaggle mount when present, otherwise
        the materialised catalog inside the managed dataset tree."""
        return self._kaggle_input_root() or self.catalog_root

    def source_roots(self) -> list[Path]:
        """Every filesystem root the provider may legitimately read from."""
        return self._allowed_roots()

    def _allowed_roots(self) -> list[Path]:
        roots = [self.catalog_root.resolve()]
        mounted = self._kaggle_input_root()
        if mounted is not None:
            roots.append(mounted.resolve())
        try:
            source = self.downloader.resolve_downloaded(self.handle)
            roots.append(source.resolve())
        except Exception:  # noqa: BLE001 - source may not be downloaded yet
            pass
        return roots

    # ------------------------------------------------------------------ #
    # Provider introspection
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def capabilities(self) -> ProviderCapabilities:
        """Declared imagery capabilities (used by the provider registry)."""
        return ProviderCapabilities(
            name=self.name,
            kind=self.kind,
            priority=100,
            features=[
                "ensure",
                "location",
                "validate",
                "catalog",
                "discover_ndvi",
                "discover_evi",
                "read_metadata",
                "read",
                "patch",
                "get_historical_context",
                "generate_metadata",
                "crs_validation",
                "resolution_validation",
                "band_validation",
            ],
        )

    def available(self) -> bool:
        try:
            location = self.location()
            return location.downloaded or location.materialized
        except Exception:  # noqa: BLE001 - availability is best-effort
            return False

    def manifest(self) -> ProviderManifest:
        location = self.location()
        details: dict[str, Any] = {"handle": self.handle}
        if location.materialized and not self._kaggle_input_root():
            details.update(
                {
                    "files": location.files,
                    "size_bytes": location.size_bytes,
                    "version": location.version,
                    "years": self.catalog().years,
                    "ndvi": len(self.catalog().ndvi),
                    "evi": len(self.catalog().evi),
                }
            )
        return ProviderManifest(
            name=self.name,
            kind=self.kind,
            status=self.status,
            available=self.available(),
            root=location.root if location.root.is_dir() else None,
            details=details,
        )

    def describe(self) -> dict[str, Any]:
        return self.manifest().to_dict()

    # ------------------------------------------------------------------ #
    # Acquisition
    # ------------------------------------------------------------------ #

    def ensure(
        self, *, force: bool = False, materialize: bool | None = None
    ) -> Path:
        """Download (or reuse) the imagery dataset and materialise it.

        Args:
            force: Re-download even when a cached copy exists.
            materialize: Mirror the download into ``catalog_root`` (defaults
                to the provider configuration).

        Returns:
            The materialised root path (``catalog_root``), the Kaggle
            cache path when materialisation is disabled, or the mounted
            input root when the dataset is attached under ``/kaggle/input``.
        """
        mounted = self._kaggle_input_root()
        if mounted is not None:
            self._status = ProviderStatus.READY
            logger.info(
                "Using Kaggle-mounted imagery dataset",
                extra={"root": str(mounted), "handle": self.handle},
            )
            return mounted

        should_materialize = self.materialize if materialize is None else materialize
        self._status = ProviderStatus.NOT_INITIALIZED
        source = self.downloader.download(
            self.handle, force=force or self.force_download
        )

        if not should_materialize:
            self._status = ProviderStatus.READY
            return source

        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.downloader.materialize(
            source,
            self.catalog_root,
            progress=lambda done, total, name: logger.debug(
                "materialising",
                extra={"done": done, "total": total, "file": name},
            ),
        )
        if self.verify_integrity and hasattr(self.downloader, "verify_integrity"):
            ok = self.downloader.verify_integrity(self.catalog_root)
            if not ok:
                self._status = ProviderStatus.ERROR
                logger.error("Integrity verification failed after download")
            else:
                self._status = ProviderStatus.READY
        else:
            self._status = ProviderStatus.READY
        self._inventory = None
        self._catalog = None
        return self.catalog_root

    def location(self) -> ImageDatasetLocation:
        """Current on-disk location / materialisation state."""
        mounted = self._kaggle_input_root()
        if mounted is not None:
            return ImageDatasetLocation(
                handle=self.handle,
                root=mounted,
                downloaded=True,
                materialized=True,
                version=_DEFAULT_VERSION,
                files=0,
                size_bytes=0,
                cache_root=self._cache_root(),
            )
        downloaded = self.downloader.is_downloaded(self.handle)
        materialized = self.catalog_root.is_dir() and any(self.catalog_root.rglob("*"))

        root = self.catalog_root if materialized else self._source_root_or_default()
        files, size_bytes = self._tree_stats(self.catalog_root if materialized else root)

        return ImageDatasetLocation(
            handle=self.handle,
            root=root,
            downloaded=downloaded,
            materialized=materialized,
            version=_DEFAULT_VERSION,
            files=files,
            size_bytes=size_bytes,
            cache_root=self._cache_root(),
        )

    def _source_root_or_default(self) -> Path:
        try:
            return self.downloader.resolve_downloaded(self.handle)
        except Exception:  # noqa: BLE001
            return self.catalog_root

    def _tree_stats(self, root: Path) -> tuple[int, int]:
        if not root.is_dir():
            return 0, 0
        files = [p for p in root.rglob("*") if p.is_file()]
        return len(files), sum(p.stat().st_size for p in files)

    def _cache_root(self) -> Path | None:
        cache = getattr(self.downloader, "_cache_root", None)
        return Path(cache) if cache is not None else None

    # ------------------------------------------------------------------ #
    # Inventory / validation / metadata
    # ------------------------------------------------------------------ #

    def _assert_materialized(self) -> Path:
        root = self.source_root()
        if not root.is_dir():
            raise DatasetNotFoundError(
                "Imagery dataset is not available. Run `ensure()` first.",
                detail=str(root),
            )
        return root

    def scan(self, *, use_cache: bool | None = None) -> DatasetInventory:
        """Scan the materialised imagery root into an inventory."""
        root = self._assert_materialized()
        if self._inventory is None or use_cache is False:
            self._inventory = self.scanner.scan(
                root, use_cache=use_cache is None or use_cache
            )
        return self._inventory

    def validate(self, *, report_dir: str | Path | None = None) -> ValidationReport:
        """Validate the materialised imagery and return a report."""
        root = self._assert_materialized()
        inventory = self.scan(use_cache=True)
        report = self.validator.validate(root, inventory)
        if report_dir is not None:
            out_dir = Path(report_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report.write_json(out_dir / "validation_report.json")
        return report

    def generate_metadata(self, *, force: bool = False) -> int:
        """Generate / refresh per-file metadata records."""
        root = self._assert_materialized()
        inventory = self.scan(use_cache=True)
        records = self.metadata_generator.generate(root, inventory, force=force)
        return len(records)

    # ------------------------------------------------------------------ #
    # Characteristic validation (CRS / resolution / bands)
    # ------------------------------------------------------------------ #

    def validate_crs(self, expected: str | None = None) -> dict[str, Any]:
        """Check that every raster shares a usable, consistent CRS.

        Args:
            expected: Optional CRS string every raster must match (e.g.
                ``"EPSG:4326"``). When None, consistency across rasters is
                checked instead.

        Returns:
            ``{valid, expected, found, distinct, issues}``.
        """
        entries = self.catalog().entries
        rasters = [e for e in entries if e.category is FileCategory.GEOTIFF]
        if not rasters:
            return {"valid": True, "expected": expected, "found": None,
                    "distinct": [], "issues": []}

        crs_of: list[str] = []
        issues: list[dict[str, Any]] = []
        for entry in rasters:
            try:
                crs = self.read_metadata(entry.path).crs
            except Exception as exc:  # noqa: BLE001 - best effort
                issues.append(
                    {"path": entry.relative_path, "code": "unreadable", "reason": str(exc)}
                )
                continue
            if not crs or not str(crs).strip():
                issues.append(
                    {"path": entry.relative_path, "code": "missing_crs"}
                )
                continue
            crs_of.append(str(crs))

        distinct = sorted({c for c in crs_of if c})
        expected_norm = str(expected).upper() if expected else None
        if expected_norm:
            valid = all(c.upper() == expected_norm for c in crs_of) and not issues
        else:
            valid = len(distinct) <= 1 and not issues
        return {
            "valid": valid,
            "expected": expected,
            "found": distinct[0] if len(distinct) == 1 else distinct,
            "distinct": distinct,
            "issues": issues,
        }

    def validate_resolution(
        self, expected: list[str] | None = None
    ) -> dict[str, Any]:
        """Check that the catalog covers the expected resolution bands.

        Args:
            expected: Resolution bands that must be present (e.g.
                ``["R10m", "R20m"]``). Defaults to the provider's configured
                expectation of both bands.

        Returns:
            ``{valid, expected, available, missing}``.
        """
        catalog = self.catalog()
        available = sorted(catalog.resolutions)
        wanted = list(expected or ["R10m", "R20m"])
        missing = sorted(set(wanted) - set(available))
        return {
            "valid": not missing,
            "expected": wanted,
            "available": available,
            "missing": missing,
        }

    def validate_bands(self, expected: int = _EXPECTED_BANDS) -> dict[str, Any]:
        """Check that every raster carries the expected band count.

        Returns:
            ``{valid, expected, found, distinct, issues}``.
        """
        entries = self.catalog().entries
        rasters = [e for e in entries if e.category is FileCategory.GEOTIFF]
        if not rasters:
            return {"valid": True, "expected": expected, "found": None,
                    "distinct": [], "issues": []}

        distinct: dict[int, int] = {}
        issues: list[dict[str, Any]] = []
        for entry in rasters:
            try:
                bands = self.read_metadata(entry.path).bands
            except Exception as exc:  # noqa: BLE001 - best effort
                issues.append(
                    {"path": entry.relative_path, "code": "unreadable", "reason": str(exc)}
                )
                continue
            distinct[bands] = distinct.get(bands, 0) + 1

        valid = all(b == expected for b in distinct) and not issues
        return {
            "valid": valid,
            "expected": expected,
            "found": list(distinct),
            "distinct": distinct,
            "issues": issues,
        }

    # ------------------------------------------------------------------ #
    # Catalog / discovery
    # ------------------------------------------------------------------ #

    def catalog(self, *, refresh: bool = False) -> ImageCatalog:
        """Classified inventory of the imagery dataset."""
        if self._catalog is not None and not refresh:
            return self._catalog
        inventory = self.scan(use_cache=not refresh)
        location = self.location()
        entries = sorted(inventory.entries, key=lambda e: str(e.path))

        ndvi = [e for e in entries if e.category is FileCategory.GEOTIFF and e.index_type is IndexType.NDVI]
        evi = [e for e in entries if e.category is FileCategory.GEOTIFF and e.index_type is IndexType.EVI]
        years = sorted({e.year for e in entries if e.year is not None})
        resolutions = sorted(
            {
                e.resolution.value
                for e in entries
                if e.resolution is not Resolution.UNKNOWN
            }
        )
        counts = inventory.counts()
        location.files = len(entries)
        location.size_bytes = inventory.total_size()

        self._catalog = ImageCatalog(
            location=location,
            entries=entries,
            ndvi=ndvi,
            evi=evi,
            years=years,
            resolutions=resolutions,
            counts=counts,
        )
        self._status = ProviderStatus.READY if entries else ProviderStatus.MISSING_DATA
        return self._catalog

    def discover_ndvi(self) -> list[FileEntry]:
        """Discover NDVI rasters (lazy — no pixel data loaded)."""
        return list(self.catalog().ndvi)

    def discover_evi(self) -> list[FileEntry]:
        """Discover EVI rasters (lazy — no pixel data loaded)."""
        return list(self.catalog().evi)

    # ------------------------------------------------------------------ #
    # Lazy raster access
    # ------------------------------------------------------------------ #

    def _resolve(self, path: str | Path, *, must_exist: bool = True) -> Path:
        candidate = Path(path).expanduser()
        roots = [root for root in self._allowed_roots()]
        if candidate.is_absolute():
            candidates = [candidate]
        else:
            candidates = [candidate.resolve()] + [root / candidate for root in roots]
        for cand in candidates:
            try:
                cand = cand.resolve()
                for root in roots:
                    cand.relative_to(root)
            except ValueError:
                continue
            if (not must_exist) or cand.exists():
                return cand
        raise DatasetNotFoundError(
            f"Raster path is missing or outside the imagery roots: {path}",
            detail={"allowed_roots": [str(r) for r in roots]},
        )

    def read_metadata(self, path: str | Path) -> RasterMetadata:
        """Header-only metadata of a raster (never loads pixel data)."""
        resolved = self._resolve(path)
        return self.image_loader.read_metadata(resolved)

    def read(
        self,
        path: str | Path,
        *,
        window: tuple[int, int, int, int] | None = None,
        band: int = 1,
    ) -> np.ndarray:
        """Read a raster band (or a bounded window)."""
        resolved = self._resolve(path)
        if window is not None:
            return self.image_loader.read_window(resolved, window=window, band=band)
        load = getattr(self.image_loader, "load", None)
        if callable(load):
            return load(resolved, band=band)
        metadata = self.read_metadata(resolved)
        return self.image_loader.read_window(
            resolved, window=(0, 0, metadata.height, metadata.width), band=band
        )

    def patch(self, request: PatchRequest) -> np.ndarray:
        """Retrieve a square patch around a geographic center point.

        The center is interpreted in the raster's CRS units (longitude /
        latitude for EPSG:4326 products). The requested square is clamped to
        the raster extent when it would exceed the image bounds. **No raster
        preprocessing is applied** — the raw windowed band is returned.
        """
        metadata = self.read_metadata(request.path)
        resolved = self._resolve(request.path)
        if metadata.pixel_size is None or metadata.bounds is None:
            raise DatasetNotFoundError(
                "Raster has no georeferencing; cannot compute a patch window",
                detail=str(request.path),
            )
        xres, yres = metadata.pixel_size
        left, _bottom, right, top = metadata.bounds
        width, height = metadata.width, metadata.height
        if width <= 0 or height <= 0 or xres <= 0 or yres <= 0:
            raise DatasetNotFoundError(
                "Raster has invalid dimensions / pixel size",
                detail=str(request.path),
            )

        center_x, center_y = request.center
        col = (center_x - left) / xres
        row = (top - center_y) / yres
        half = max(1, request.size // 2)
        col0 = int(col - half)
        row0 = int(row - half)

        # Clamp so the window stays inside the raster (square when possible).
        col0 = max(0, min(col0, width - 1))
        row0 = max(0, min(row0, height - 1))
        w = min(request.size, width - col0)
        h = min(request.size, height - row0)
        if w <= 0 or h <= 0:
            raise DatasetNotFoundError(
                "Patch window is outside the raster extent",
                detail=str(request.path),
            )
        return self.image_loader.read_window(
            resolved,
            window=(row0, col0, h, w),
            band=request.band,
        )

    # ------------------------------------------------------------------ #
    # Historical context
    # ------------------------------------------------------------------ #

    def get_historical_context(
        self,
        *,
        window_months: list[int] | None = None,
        index_type: str | None = None,
        resolution: str | None = None,
        years: list[int] | None = None,
    ) -> HistoricalContext:
        """Per-year satellite availability for a recurring season window.

        Uses the persisted metadata store when it has records, otherwise falls
        back to classifying the scanned inventory on the fly.
        """
        months: set[int] | None = None
        if window_months is not None:
            months = {int(m) for m in window_months}

        records = self._records(
            index_type=index_type, resolution=resolution, years=years, months=months
        )

        per_year: dict[int, dict[str, int]] = {}
        for record in records:
            year = record["year"]
            obs_date = record["observation_date"]
            if year is None or obs_date is None:
                continue
            if months is not None and obs_date.month not in months:
                continue
            if years is not None and year not in set(years):
                continue
            bucket = per_year.setdefault(year, {})
            bucket["records"] = bucket.get("records", 0) + 1
            key = record["index_type"].lower()
            bucket[key] = bucket.get(key, 0) + 1

        return HistoricalContext(
            window_months=sorted(months) if months is not None else None,
            years=sorted(per_year),
            per_year=per_year,
            total_records=len(records),
            generated_at=datetime.now(),
        )

    def _records(
        self,
        *,
        index_type: str | None,
        resolution: str | None,
        years: list[int] | None,
        months: set[int] | None,
    ) -> list[dict[str, Any]]:
        """Load geotiff records from the metadata store, or build them from
        the scanned inventory when the store is empty."""
        stored: list[MetadataRecord] | None = None
        if self.metadata_store is not None:
            try:
                stored = self.metadata_store.query(
                    category="geotiff",
                    index_type=index_type,
                    resolution=resolution,
                )
            except Exception:  # noqa: BLE001 - fall back to inventory
                stored = None

        if stored:
            out: list[dict[str, Any]] = []
            for record in stored:
                if record.year is None or record.observation_date is None:
                    continue
                if index_type and record.index_type.value.upper() != str(index_type).upper():
                    continue
                if resolution and record.resolution.value != str(resolution):
                    continue
                out.append(
                    {
                        "year": record.year,
                        "observation_date": record.observation_date,
                        "index_type": record.index_type.value,
                    }
                )
            return out

        inventory = self.scan(use_cache=True)
        out = []
        for entry in inventory.entries:
            if entry.category is not FileCategory.GEOTIFF:
                continue
            if index_type and entry.index_type.value.upper() != str(index_type).upper():
                continue
            if resolution and entry.resolution.value != str(resolution):
                continue
            out.append(
                {
                    "year": entry.year,
                    "observation_date": parse_observation_date(entry.path),
                    "index_type": entry.index_type.value,
                }
            )
        return out
