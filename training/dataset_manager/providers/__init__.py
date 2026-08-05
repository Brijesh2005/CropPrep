"""Dataset Manager data providers.

The provider layer is the **only** place the Dataset Manager talks to data
sources. Every provider implements one of the two ports declared in
:mod:`~training.dataset_manager.providers.base`:

* :class:`TabularProvider` — structured (CSV) datasets. The concrete
  implementation :class:`GitRepositoryTabularProvider` serves the Git-versioned
  CSVs under ``training/datasets/tabular/`` (auto-discovery, schema
  validation, joins, streaming, missing-value handling, metadata).
* :class:`ImageProvider` — remote-sensing imagery. The concrete implementation
  :class:`KaggleHubImageProvider` serves the Kaggle Sentinel-2 NDVI / EVI
  GeoTIFFs (download-or-reuse, validation, lazy raster access, patch
  retrieval, historical context).

Dependency rules (must hold at all times)::

    DatasetManager --> providers (ports)
    providers      --> independent sources (Git repo, Kaggle)

Providers never import the Dataset Manager and never import each other, so
the dependency graph stays acyclic.
"""

from __future__ import annotations

from ..logger import get_logger as _get_logger
from .base import ImageProvider, Provider, TabularProvider
from .models import (
    ImageCatalog,
    ImageDatasetLocation,
    PatchRequest,
    ProviderManifest,
    ProviderStatus,
    TabularCatalog,
    TabularDatasetInfo,
    TabularJoinSpec,
)

logger = _get_logger("providers")

__all__ = [
    "Provider",
    "TabularProvider",
    "ImageProvider",
    "GitRepositoryTabularProvider",
    "KaggleHubImageProvider",
    "ProviderStatus",
    "ProviderManifest",
    "TabularDatasetInfo",
    "TabularCatalog",
    "TabularJoinSpec",
    "ImageDatasetLocation",
    "ImageCatalog",
    "PatchRequest",
]


def __getattr__(name: str) -> object:
    """Lazy-import the concrete providers to avoid heavy imports at package
    load time (rasterio, pandas, kagglehub are imported by the engines they
    wrap)."""
    if name == "GitRepositoryTabularProvider":
        from .git_tabular import GitRepositoryTabularProvider

        return GitRepositoryTabularProvider
    if name == "KaggleHubImageProvider":
        from .kaggle_image import KaggleHubImageProvider

        return KaggleHubImageProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
