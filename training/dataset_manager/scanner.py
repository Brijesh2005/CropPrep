"""Directory scanner that builds a :class:`DatasetInventory`.

The scanner is the only module responsible for *discovery*: it walks the
dataset tree, classifies every file (CSV / GeoTIFF / other), tags rasters
with index type (NDVI/EVI), resolution band (R10m/R20m) and year, and
assembles the result into a :class:`DatasetInventory`.

Performance considerations:

* **Parallel scanning** — file classification / stat runs on a thread pool
  (:func:`~services.dataset_manager.utils.run_parallel`).
* **Cache-aware** — a cheap tree signature ``(file_count, total_size,
  max_mtime)`` is stored with the inventory. When the signature is unchanged
  the inventory is served from the cache instead of re-walking the tree.
* **Lazy hashing** — content hashes are only computed when ``hash_files`` is
  enabled (default off) because hashing is the most expensive step.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Iterable

from .cache_manager import CacheManager
from .config import ScanConfig
from .exceptions import DatasetNotFoundError, ScannerError
from .interfaces import Cache, Scanner
from .logger import get_logger
from .models import DatasetInventory, FileCategory, FileEntry
from .utils import (
    DEFAULT_EXCLUDE_DIRS,
    classify_index_type_from_path,
    classify_resolution_from_path,
    extract_year_from_path,
    is_csv_path,
    is_geotiff_path,
    run_parallel,
    sha256_file,
    tree_signature,
    walk_files,
)

logger = get_logger("scanner")

#: File extensions treated as rasters (magic-byte detection is preferred but
#: extension is used to classify before validation flags the broken file).
_RASTER_SUFFIXES = {".tif", ".tiff"}


class DatasetScanner(Scanner):
    """Concrete :class:`Scanner` implementation.

    Args:
        config: Scan configuration section.
        cache: Optional :class:`Cache` used to memoise inventories. When
            omitted a fresh in-memory cache is used.
    """

    def __init__(
        self,
        config: ScanConfig | None = None,
        *,
        cache: Cache | None = None,
        exclude_dirs: Iterable[str] | None = None,
    ) -> None:
        self.config = config or ScanConfig()
        self.cache = cache or CacheManager(enabled=False)
        self.exclude_dirs = (
            frozenset(exclude_dirs) if exclude_dirs is not None else DEFAULT_EXCLUDE_DIRS
        )

    def scan(self, root: Path, *, use_cache: bool = True) -> DatasetInventory:
        """Scan ``root`` and return a classified inventory.

        Args:
            root: Dataset directory to scan.
            use_cache: Serve from / populate the scan cache.

        Returns:
            The inventory.

        Raises:
            DatasetNotFoundError: When ``root`` does not exist.
            ScannerError: When scanning fails unexpectedly.
        """
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise DatasetNotFoundError(
                f"Dataset root does not exist: {root}", detail=str(root)
            )

        cache_key = f"scan:{root}"
        if use_cache and self.config.use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                # Invalidate automatically when the tree changed (cheap
                # signature: file count / total size / newest mtime).
                cached_signature = list(cached.get("tree_signature", []))
                current_signature = list(
                    tree_signature(root, exclude_dirs=self.exclude_dirs)
                )
                if cached_signature == current_signature:
                    inventory = DatasetInventory.from_dict(cached)
                    inventory.source = "cache"
                    logger.debug(
                        "Serving inventory from cache",
                        extra={"root": str(root), "files": len(inventory.entries)},
                    )
                    return inventory
                self.cache.delete(cache_key)
                logger.debug(
                    "Scan cache invalidated (tree changed)",
                    extra={"root": str(root)},
                )

        started = time.perf_counter()
        try:
            files = list(walk_files(root, exclude_dirs=self.exclude_dirs))
            classify = partial(self._classify_file, root=root)
            entries = run_parallel(files, classify, workers=self.config.workers)
        except OSError as exc:
            raise ScannerError(f"Failed to scan {root}: {exc}", detail=str(exc)) from exc

        clean_entries = [e for e in entries if isinstance(e, FileEntry)]
        inventory = DatasetInventory(
            root=root,
            entries=clean_entries,
            duration_s=time.perf_counter() - started,
            source="scan",
        )

        if use_cache and self.config.use_cache:
            payload = inventory.to_dict()
            payload["tree_signature"] = list(
                tree_signature(root, exclude_dirs=self.exclude_dirs)
            )
            self.cache.set(cache_key, payload)

        logger.info(
            "Scan complete",
            extra={
                "root": str(root),
                "files": len(clean_entries),
                "duration_s": round(inventory.duration_s, 3),
            },
        )
        return inventory

    def invalidate(self, root: Path) -> int:
        """Drop cached inventories for ``root``; returns the count removed."""
        return self.cache.delete_prefix(f"scan:{root.resolve()}")

    # -- Internals ------------------------------------------------------------- #

    def _classify_file(self, path: Path, *, root: Path) -> FileEntry:
        """Classify a single file into a :class:`FileEntry`."""
        stat = path.stat()
        dotted_suffix = path.suffix.lower()          # e.g. ".tif", ".csv"
        extension = dotted_suffix.lstrip(".")        # e.g. "tif", "csv"

        if is_csv_path(path):
            category = FileCategory.CSV
        elif dotted_suffix in _RASTER_SUFFIXES:
            category = FileCategory.GEOTIFF
        else:
            category = FileCategory.OTHER

        entry = FileEntry(
            path=path,
            relative_path=path.relative_to(root).as_posix(),
            category=category,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            extension=extension,
        )

        if category is FileCategory.GEOTIFF:
            entry.index_type = classify_index_type_from_path(path)
            entry.resolution = classify_resolution_from_path(path)

        # The calendar year is extracted for every file (CSVs too), since
        # tabular records carry a year as well.
        entry.year = extract_year_from_path(path)

        if self.config.hash_files:
            entry.sha256 = sha256_file(path)

        return entry
