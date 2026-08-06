"""Provider pattern: abstract interfaces for the Dataset Manager.

The Dataset Manager **never reads files directly**. Every data source is
wrapped in a *provider* that implements one of the two ports declared here:

* :class:`TabularProvider` — structured (CSV) datasets. Implementations
  include :class:`~training.dataset_manager.providers.git_tabular.
  GitRepositoryTabularProvider` (Git-versioned CSVs).
* :class:`ImageProvider` — remote-sensing imagery. Implementations include
  :class:`~training.dataset_manager.providers.kaggle_image.
  KaggleHubImageProvider` (Kaggle Sentinel-2 NDVI / EVI GeoTIFFs).

Both extend :class:`Provider`, the common introspection contract (status,
availability, manifest). Providers are **independent** — they never import the
Dataset Manager and never import each other, which keeps the dependency rules
acyclic:

    DatasetManager --> TabularProvider --> source
    DatasetManager --> ImageProvider --> source
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from ..models import HistoricalContext, RasterMetadata, ValidationReport
from .models import (
    ImageCatalog,
    ImageDatasetLocation,
    PatchRequest,
    ProviderCapabilities,
    ProviderHealth,
    ProviderManifest,
    ProviderStatus,
    TabularCatalog,
    TabularJoinSpec,
)


class Provider(ABC):
    """Common contract for every data provider.

    Subclasses declare a stable ``name`` and ``kind`` and expose a manifest
    used by the bootstrap, the CLI and diagnostics. The concrete
    :meth:`capabilities` / :meth:`health` defaults are derived from the
    manifest and may be overridden by implementations to declare richer
    feature sets.
    """

    name: str = "provider"
    kind: str = "generic"

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        """Current provider lifecycle status."""

    @abstractmethod
    def available(self) -> bool:
        """True when the provider can serve its data right now."""

    @abstractmethod
    def manifest(self) -> ProviderManifest:
        """Structured description of this provider instance."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Human + machine readable provider summary."""

    def capabilities(self) -> ProviderCapabilities:
        """Declared capabilities of this provider (concrete default).

        Override to declare the feature set (e.g. ``["discover", "join"]``)
        used by the registry to answer capability queries.
        """
        return ProviderCapabilities(
            name=self.name,
            kind=self.kind,
            features=["manifest", "describe"],
        )

    def health(self) -> ProviderHealth:
        """Snapshot of current provider health (concrete default)."""
        import time

        start = time.perf_counter()
        available = self.available()
        latency = time.perf_counter() - start
        return ProviderHealth(
            name=self.name,
            kind=self.kind,
            status=self.status,
            available=available,
            latency_s=latency,
            detail=self.manifest().to_dict().get("details", {}),
        )


class TabularProvider(Provider):
    """Port for structured (tabular) datasets.

    Implementations discover datasets *automatically* under a configured root
    — no filenames are ever hardcoded. Consumers address datasets by their
    discovered ``name``.
    """

    kind = "tabular"

    @abstractmethod
    def discover(self, *, refresh: bool = False) -> TabularCatalog:
        """Discover all tabular datasets under the provider root."""

    @abstractmethod
    def names(self) -> list[str]:
        """Discovered dataset names (sorted)."""

    @abstractmethod
    def load(
        self,
        name: str,
        *,
        chunksize: int | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load a dataset as a DataFrame (or an iterator when chunked)."""

    @abstractmethod
    def stream(self, name: str, chunksize: int, **kwargs: Any) -> Iterator[Any]:
        """Stream a dataset in bounded chunks (memory bounded)."""

    @abstractmethod
    def schema(self, name: str) -> dict[str, Any]:
        """Schema / dtype / missing-value profile of a dataset."""

    @abstractmethod
    def validate_schema(self, name: str) -> dict[str, Any]:
        """Run schema validation; returns ``{valid, issues}``."""

    @abstractmethod
    def statistics(self, name: str) -> dict[str, Any]:
        """Numeric column statistics (mean / std / min / max / count)."""

    @abstractmethod
    def missing_values(self, name: str) -> dict[str, int]:
        """``{column: missing_count}`` for a dataset."""

    @abstractmethod
    def handle_missing(
        self,
        name: str,
        strategy: str = "drop",
        *,
        fill_value: Any = None,
        fill_method: str = "mean",
    ) -> Any:
        """Return a copy of the dataset with missing values handled.

        ``strategy``: ``drop`` (drop rows with any missing) or ``fill``
        (fill with ``fill_value`` or a per-column ``fill_method`` of
        ``mean`` / ``median`` / ``mode`` / ``constant``).
        """

    @abstractmethod
    def join(self, joins: list[TabularJoinSpec], *, how: str | None = None) -> Any:
        """Sequentially join discovered datasets into a single frame."""

    @abstractmethod
    def metadata(self, name: str) -> dict[str, Any]:
        """Dataset metadata (path, size, schema, statistics)."""


class ImageProvider(Provider):
    """Port for remote-sensing imagery datasets.

    Implementations acquire the dataset (download or reuse), validate it,
    classify NDVI / EVI rasters and expose **lazy** raster access — full
    rasters are never loaded implicitly, and no raster preprocessing is done
    at this layer.
    """

    kind = "image"

    @abstractmethod
    def ensure(self, *, force: bool = False, materialize: bool | None = None) -> Path:
        """Download (or reuse) the imagery dataset; return its root path.

        ``force=True`` triggers a fresh download. When ``materialize`` is set
        the dataset is mirrored into the provider's managed root.
        """

    @abstractmethod
    def location(self) -> ImageDatasetLocation:
        """Current on-disk location / materialisation state."""

    @abstractmethod
    def validate(self, *, report_dir: str | Path | None = None) -> ValidationReport:
        """Validate the downloaded imagery and return a report."""

    @abstractmethod
    def catalog(self, *, refresh: bool = False) -> ImageCatalog:
        """Classified inventory of the imagery (NDVI / EVI, year, resolution)."""

    @abstractmethod
    def discover_ndvi(self) -> list[Any]:
        """Discover NDVI rasters (lazy metadata / file entries)."""

    @abstractmethod
    def discover_evi(self) -> list[Any]:
        """Discover EVI rasters (lazy metadata / file entries)."""

    @abstractmethod
    def read_metadata(self, path: str | Path) -> RasterMetadata:
        """Header-only metadata of a raster (never loads pixel data)."""

    @abstractmethod
    def read(
        self,
        path: str | Path,
        *,
        window: tuple[int, int, int, int] | None = None,
        band: int = 1,
    ) -> Any:
        """Read a raster band (or a bounded window) into an array."""

    @abstractmethod
    def patch(self, request: PatchRequest) -> Any:
        """Retrieve a square patch of ``request.size`` around a geographic
        center point (patch retrieval interface — no preprocessing)."""

    @abstractmethod
    def get_historical_context(
        self,
        *,
        window_months: list[int] | None = None,
        index_type: str | None = None,
        resolution: str | None = None,
        years: list[int] | None = None,
    ) -> HistoricalContext:
        """Temporal availability of image records for a season window."""

    @abstractmethod
    def generate_metadata(self, *, force: bool = False) -> int:
        """Generate / refresh metadata records for every discovered file."""
