"""Ports (abstract interfaces) for the Spatial-Temporal Alignment Module.

STAM follows the same hexagonal style as the Dataset Manager: the concrete
orchestrator (:class:`~services.spatial_alignment.stam.STAM`) depends on these
ports, and the production implementations wrap the Dataset Manager — the only
allowed data access path. Tests inject fakes to isolate each concern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import numpy as np

from .observation import ImageRecordRef


class ImageMetadataSource(ABC):
    """Retrieve image metadata records through the Dataset Manager."""

    @abstractmethod
    def query_images(
        self,
        *,
        index_type: str | None = None,
        resolution: str | None = None,
        year: int | None = None,
    ) -> list[ImageRecordRef]:
        """Return image records matching the filters (index/resolution/year)."""

    @abstractmethod
    def image_metadata(self, path: str) -> ImageRecordRef:
        """Fetch the full metadata record for a single image path."""


class ImageReader(ABC):
    """Lazy windowed access to raster pixels."""

    @abstractmethod
    def read_window(
        self, path: str, window: tuple[int, int, int, int], band: int = 1
    ) -> np.ndarray:
        """Read ``(row_off, col_off, height, width)`` of a band."""

    @abstractmethod
    def read_metadata(self, path: str) -> dict[str, Any]:
        """Header-only metadata for a raster path."""


class TabularSource(ABC):
    """Match tabular agricultural records for a location/season/year."""

    @abstractmethod
    def load_record(
        self,
        *,
        village: str | None,
        district: str | None,
        year: int,
        season: str | None,
    ) -> dict[str, Any] | None:
        """Return the best matching tabular record, or None."""

    @abstractmethod
    def available_years(self) -> list[int]:
        """Sorted distinct years present in the tabular table."""


class LocationCatalog(ABC):
    """Source of dataset locations used to build the spatial index."""

    @abstractmethod
    def points(self) -> list[dict[str, Any]]:
        """Return ``[{id, name, lon, lat, meta}]`` dataset locations."""


class BoundaryProvider(ABC):
    """Provide administrative boundary geometries (via the Dataset Manager)."""

    @abstractmethod
    def boundaries(self) -> Any:
        """Return a GeoDataFrame of admin boundaries (EPSG:4326)."""


class StamCache(ABC):
    """Namespaced key/value cache for spatial/temporal/observation results."""

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def clear(self, prefix: str | None = None) -> int: ...
