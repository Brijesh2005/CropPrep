"""Geographic patch extraction from Sentinel imagery.

:class:`PatchExtractor` turns a **farmer location** (latitude / longitude in
WGS84) and a patch size into a raw NumPy patch. It composes the existing image
provider capabilities:

1. **Locate** — pick the best raster for the requested index type /
   resolution / year (nearest raster center to the requested point).
2. **Convert** — transform the WGS84 point into the raster's CRS (pyproj).
3. **Extract** — windowed read through :meth:`ImageProvider.patch` (lazy,
   bounded memory — the full raster is never loaded).
4. **Pad** — optionally pad edge patches to the exact requested size.
5. **Record** — persist :class:`PatchMetadata` in the extended metadata DB.

**No raster preprocessing is performed** — the raw band window is returned.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .exceptions import DatasetNotFoundError
from .interfaces import PatchExtractor
from .logger import get_logger
from .metadata_repository import MetadataRepository
from .models import FileCategory, IndexType, PatchMetadata, Resolution
from .providers.base import ImageProvider
from .utils import safe_float

logger = get_logger("patch_extractor")


class PatchExtractorImpl(PatchExtractor):
    """Concrete :class:`PatchExtractor` backed by an image provider.

    Args:
        image_provider: The imagery provider (usually resolved through the
            provider registry).
        metadata_repository: Optional :class:`MetadataRepository` used to
            persist :class:`PatchMetadata` records for auditability.
        max_candidates: Bounded candidate inspection per extraction.
    """

    def __init__(
        self,
        image_provider: ImageProvider,
        *,
        metadata_repository: MetadataRepository | None = None,
        max_candidates: int = 64,
    ) -> None:
        self.image_provider = image_provider
        self.metadata_repository = metadata_repository
        self.max_candidates = max_candidates

    # -- Public API ------------------------------------------------------------ #

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
    ) -> np.ndarray:
        """Extract a ``size`` x ``size`` patch around ``(latitude, longitude)``.

        Args:
            latitude / longitude: WGS84 coordinates of the location.
            size: Square patch edge length in pixels.
            index_type: ``"NDVI"`` / ``"EVI"`` (None for either).
            resolution: ``"R10m"`` / ``"R20m"`` (None for either).
            year: Restrict to a specific year (None for any).
            band: 1-based band index (default 1).
            padding: Pad edge patches to exactly ``size``.

        Returns:
            A raw :class:`numpy.ndarray` of shape ``(size, size)``.

        Raises:
            DatasetNotFoundError: When no raster matches the request or the
                patch falls entirely outside the raster extent.
        """
        array, _metadata = self._extract(
            latitude, longitude, size,
            index_type=index_type, resolution=resolution, year=year,
            band=band, padding=padding,
        )
        return array

    def extract_with_metadata(
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
    ) -> tuple[np.ndarray, PatchMetadata]:
        """Like :meth:`extract` but also returns the :class:`PatchMetadata`."""
        return self._extract(
            latitude, longitude, size,
            index_type=index_type, resolution=resolution, year=year,
            band=band, padding=padding,
        )

    # -- Internals ------------------------------------------------------------- #

    def _extract(
        self,
        latitude: float,
        longitude: float,
        size: int,
        *,
        index_type: str | None,
        resolution: str | None,
        year: int | None,
        band: int,
        padding: bool,
    ) -> tuple[np.ndarray, PatchMetadata]:
        if size <= 0:
            raise ValueError(f"Patch size must be positive, got {size}")
        lat = safe_float(latitude)
        lon = safe_float(longitude)
        if lat is None or lon is None:
            raise ValueError(
                f"Invalid coordinates: latitude={latitude!r}, longitude={longitude!r}"
            )
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError(
                f"Coordinates must be finite: latitude={lat!r}, longitude={lon!r}"
            )

        entry = self._locate_raster(
            latitude=lat, longitude=lon,
            index_type=index_type, resolution=resolution, year=year,
        )
        meta = self.image_provider.read_metadata(entry.path)
        center = self._to_raster_crs(lat, lon, meta.crs)

        from .providers.models import PatchRequest

        try:
            patch = self.image_provider.patch(
                PatchRequest(path=entry.path, center=center, size=size, band=band)
            )
        except DatasetNotFoundError as exc:
            raise DatasetNotFoundError(
                f"No raster covers the location ({lat}, {lon}) for the requested patch",
                detail={"path": str(entry.path), "reason": exc.message},
            ) from exc

        padded = False
        array = np.asarray(patch, dtype=patch.dtype)
        if padding and (array.shape[0] != size or array.shape[1] != size):
            pad_h = max(0, size - array.shape[0])
            pad_w = max(0, size - array.shape[1])
            array = np.pad(
                array,
                ((0, pad_h), (0, pad_w)),
                mode="edge",
            )
            padded = True

        metadata = PatchMetadata(
            path=entry.path,
            center=center,
            size=size,
            band=band,
            crs=meta.crs,
            resolution=meta.resolution.value,
            padded=padded,
        )
        if self.metadata_repository is not None:
            self.metadata_repository.save_patch(metadata)
        return array, metadata

    def _locate_raster(
        self,
        *,
        latitude: float,
        longitude: float,
        index_type: str | None,
        resolution: str | None,
        year: int | None,
    ) -> Any:
        """Pick the raster best matching the request (nearest center wins)."""
        index = _normalise_index(index_type)
        if index is None:
            candidates = list(self.image_provider.discover_ndvi()) + list(
                self.image_provider.discover_evi()
            )
        elif index is IndexType.NDVI:
            candidates = list(self.image_provider.discover_ndvi())
        else:
            candidates = list(self.image_provider.discover_evi())

        wanted_res = _normalise_resolution(resolution)
        candidates = [
            e
            for e in candidates
            if e.category is FileCategory.GEOTIFF
            and (wanted_res is None or e.resolution is wanted_res)
            and (year is None or e.year == year)
        ]
        if not candidates:
            raise DatasetNotFoundError(
                "No imagery matches the patch request",
                detail={
                    "index_type": index_type,
                    "resolution": resolution,
                    "year": year,
                    "available_years": self.image_provider.catalog().years,
                    "available_resolutions": self.image_provider.catalog().resolutions,
                },
            )

        candidates = candidates[: self.max_candidates]
        best: tuple[float, Any] | None = None
        for entry in candidates:
            meta = self.image_provider.read_metadata(entry.path)
            if meta.bounds is None:
                continue
            left, bottom, right, top = meta.bounds
            center_x = (left + right) / 2.0
            center_y = (bottom + top) / 2.0
            distance = abs(center_x - longitude) + abs(center_y - latitude)
            if best is None or distance < best[0]:
                best = (distance, entry)

        if best is None:
            raise DatasetNotFoundError(
                "No candidate raster has usable georeferencing",
                detail={"index_type": index_type, "year": year},
            )
        return best[1]

    @staticmethod
    def _to_raster_crs(latitude: float, longitude: float, crs: str | None) -> tuple[float, float]:
        """Convert a WGS84 point into the raster CRS (identity for EPSG:4326).

        Rasterio's transform expects coordinates in the raster's CRS; Sentinel
        NDVI/EVI products in this project are EPSG:4326, but UTM products are
        supported transparently via pyproj.
        """
        if crs is None:
            return (longitude, latitude)
        upper = str(crs).upper()
        if "4326" in upper or "WGS84" in upper:
            return (longitude, latitude)
        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
            x, y = transformer.transform(longitude, latitude)
            return (float(x), float(y))
        except Exception:  # noqa: BLE001 - fall back to identity
            return (longitude, latitude)


def _normalise_index(index_type: str | None) -> IndexType | None:
    if index_type is None:
        return None
    upper = str(index_type).upper()
    if upper == "NDVI":
        return IndexType.NDVI
    if upper == "EVI":
        return IndexType.EVI
    raise ValueError(f"Unsupported index type: {index_type} (expected NDVI/EVI)")


def _normalise_resolution(resolution: str | None) -> Resolution | None:
    if resolution is None:
        return None
    upper = str(resolution).upper().replace(" ", "")
    if upper in {"R10M", "10M"}:
        return Resolution.R10M
    if upper in {"R20M", "20M"}:
        return Resolution.R20M
    raise ValueError(f"Unsupported resolution: {resolution} (expected R10m/R20m)")
