"""CropFusion Dataset Management System (DMS).

The DMS is the **single access point** for all datasets in the CropFusion
platform. Every other module (AI, GIS, backend, frontend) must communicate
only with this package — no other code reads CSVs or GeoTIFFs directly.

Public surface (see :class:`~services.dataset_manager.manager.DatasetManager`):

* **Pipeline** — :meth:`DatasetManager.download`, ``scan``, ``validate``,
  ``generate_metadata``.
* **Discovery** — :meth:`DatasetManager.inventory`, ``summary``.
* **Data access** — :meth:`DatasetManager.load_csv`, ``load_image``,
  ``list_csvs``, ``list_images`` (the only read paths).
* **Metadata** — :meth:`DatasetManager.get_metadata`, ``query_metadata``,
  ``export_metadata_parquet``.
* **Lifecycle** — :meth:`DatasetManager.register`, ``bump_version``,
  ``rollback_version``, ``registry_entries``.
* **Cache** — :meth:`DatasetManager.cache_get`, ``cache_set``,
  ``cache_invalidate``.

Example::

    from training.dataset_manager import DatasetManager

    manager = DatasetManager.from_config()
    manager.download()
    report = manager.validate()
    manager.generate_metadata()
    summary = manager.summary()
"""

from __future__ import annotations

from .cache_manager import CacheManager
from .config import Settings, load_settings
from .csv_loader import PandasCSVLoader
from .dataset_registry import SQLiteRegistry
from .downloader import KaggleDownloader
from .exceptions import (
    CacheError,
    CorruptedDatasetError,
    DatasetManagerError,
    DatasetNotFoundError,
    DownloadFailedError,
    InvalidConfigurationError,
    InvalidMetadataError,
    RegistryError,
    ScannerError,
    UnsupportedFormatError,
    ValidationFailedError,
)
from .image_loader import RasterioImageLoader
from .manager import DatasetManager
from .metadata import MetadataGeneratorImpl, SQLiteMetadataStore
from .models import (
    CSVProfile,
    DatasetInventory,
    DatasetStatus,
    DatasetSummary,
    FileCategory,
    FileEntry,
    HistoricalContext,
    IndexType,
    MetadataRecord,
    RasterMetadata,
    Resolution,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from .scanner import DatasetScanner
from .validator import DatasetValidator
from .version_manager import SQLiteVersionManager

__version__ = "0.1.0"

__all__ = [
    "DatasetManager",
    "DatasetScanner",
    "DatasetValidator",
    "KaggleDownloader",
    "PandasCSVLoader",
    "RasterioImageLoader",
    "CacheManager",
    "SQLiteRegistry",
    "SQLiteMetadataStore",
    "SQLiteVersionManager",
    "MetadataGeneratorImpl",
    "Settings",
    "load_settings",
    # Models
    "CSVProfile",
    "DatasetInventory",
    "DatasetStatus",
    "DatasetSummary",
    "FileCategory",
    "FileEntry",
    "HistoricalContext",
    "IndexType",
    "MetadataRecord",
    "RasterMetadata",
    "Resolution",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    # Exceptions
    "CacheError",
    "CorruptedDatasetError",
    "DatasetManagerError",
    "DatasetNotFoundError",
    "DownloadFailedError",
    "InvalidConfigurationError",
    "InvalidMetadataError",
    "RegistryError",
    "ScannerError",
    "UnsupportedFormatError",
    "ValidationFailedError",
]
