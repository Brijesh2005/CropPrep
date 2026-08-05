"""Provider ports: data acquisition and access abstractions.

These are the platform-agnostic ports for data providers.  The concrete
implementations live inside each platform (``training.dataset_manager.providers``
etc.) and depend on these ports rather than the reverse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from ..enums import ProviderType


class Provider(ABC):
    """Base port for any data provider."""

    #: Stable provider name, e.g. ``"kaggle-image"``.
    name: str = "provider"

    #: Kind of data this provider serves.
    provider_type: ProviderType = ProviderType.UNKNOWN

    @abstractmethod
    def health(self) -> bool:
        """Return True when the provider can serve data right now."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a machine-readable description of the provider."""


class DatasetProvider(Provider):
    """Port for providers that serve a whole dataset (catalog)."""

    @abstractmethod
    def fetch(self, *, force: bool = False) -> Path:
        """Ensure the dataset is available locally and return its root path."""

    @abstractmethod
    def exists(self) -> bool:
        """True when the dataset is already materialised locally."""

    @abstractmethod
    def version(self) -> str | None:
        """Return the current dataset version string, or None."""


class TabularProvider(Provider):
    """Port for tabular (CSV) data sources."""

    provider_type = ProviderType.TABULAR

    @abstractmethod
    def discover(self, root: Path | None = None) -> list[Path]:
        """Return every CSV file served by this provider."""

    @abstractmethod
    def load(self, path: Path, *, chunksize: int | None = None, **kwargs: Any) -> Any:
        """Load a CSV into a data frame (or an iterator when chunked)."""

    @abstractmethod
    def preview(self, path: Path, n_rows: int = 5) -> Any:
        """Return the first ``n_rows`` rows as a data frame / records."""


class ImageProvider(Provider):
    """Port for image / raster data sources."""

    provider_type = ProviderType.IMAGE

    @abstractmethod
    def catalog(self) -> list[dict[str, Any]]:
        """Return metadata for every image record served by this provider."""

    @abstractmethod
    def read_metadata(self, path: Path) -> dict[str, Any]:
        """Read header-only raster metadata without loading raster data."""

    @abstractmethod
    def read_window(
        self, path: Path, *, window: tuple[int, int, int, int], band: int = 1
    ) -> Any:
        """Read a bounded window of a band ``(row_off, col_off, height, width)``."""

    @abstractmethod
    def iterate(
        self, *, index_type: str | None = None, resolution: str | None = None
    ) -> Iterator[Path]:
        """Yield raster paths matching optional index / resolution filters."""
