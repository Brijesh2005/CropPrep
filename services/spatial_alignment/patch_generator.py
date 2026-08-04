"""Fixed-size spatial patch extraction from rasters.

Given a ``(lon, lat)`` point, :class:`SpatialPatchGenerator` produces a
square patch of configurable size (e.g. 128/224/256 px) centred on that
point. The point is first transformed into the raster's CRS, so patches stay
axis-aligned with the raster grid — no image reprojection is needed in the
common case (a point-reprojection into raster CRS is far cheaper).

Edge handling:

* **Edge correction** — when the requested window overhangs the raster, the
  available portion is read and padded to the requested size.
* **Padding** — ``constant`` (fill value, default 0) or ``reflect``.
* A validity :attr:`RasterPatch.mask` marks real vs padded pixels so Phase 4
  can weight/crop appropriately.

Requires a north-up affine (the norm for Sentinel-2 tiles). Rotated rasters
cannot be reconstructed from their bounding box and are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rasterio.transform import Affine

from .coordinate_transform import (
    geographic_to_raster_index,
    patch_window,
    window_bounds,
)
from .exceptions import PatchOutOfBoundsError
from .interfaces import ImageMetadataSource, ImageReader
from .logger import get_logger
from .observation import ImageRecordRef

logger = get_logger("patch_generator")


@dataclass(slots=True)
class RasterPatch:
    """A fixed-size raster patch centred on a geographic point."""

    path: str
    array: np.ndarray  # shape (size, size), dtype = raster dtype
    mask: np.ndarray  # bool, True where data is real (not padding)
    requested_size: int
    window: tuple[int, int, int, int]  # (row_off, col_off, height, width) read
    bounds: tuple[float, float, float, float]  # left, bottom, right, top
    crs: str | None
    resolution: tuple[float, float]
    center_lon: float
    center_lat: float
    padded: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    @property
    def valid_ratio(self) -> float:
        total = self.mask.size
        return float(self.mask.sum()) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.array.shape),
            "window": list(self.window),
            "bounds": list(self.bounds),
            "crs": self.crs,
            "resolution": list(self.resolution),
            "center_lon": self.center_lon,
            "center_lat": self.center_lat,
            "padded": self.padded,
            "valid_ratio": self.valid_ratio,
            "extra": self.extra,
        }


class SpatialPatchGenerator:
    """Concrete patch generator.

    Args:
        image_reader: :class:`ImageReader` used to read pixel windows.
        metadata_source: :class:`ImageMetadataSource` used to resolve CRS,
            transform, dimensions and resolution of the source raster.
        default_size: Patch edge length used when a call omits ``size``.
        default_pad_mode: ``"constant"`` or ``"reflect"``.
        default_pad_value: Fill value for constant padding.
        edge_correction: Shrink-then-pad at raster edges instead of failing.
    """

    def __init__(
        self,
        image_reader: ImageReader,
        metadata_source: ImageMetadataSource,
        *,
        default_size: int = 128,
        default_pad_mode: str = "constant",
        default_pad_value: float = 0.0,
        edge_correction: bool = True,
    ) -> None:
        self.reader = image_reader
        self.metadata_source = metadata_source
        self.default_size = default_size
        self.default_pad_mode = default_pad_mode
        self.default_pad_value = default_pad_value
        self.edge_correction = edge_correction

    # -- Public API ------------------------------------------------------------ #

    def get_patch(
        self,
        path: str,
        lon: float,
        lat: float,
        *,
        size: int | None = None,
        pad_mode: str | None = None,
        pad_value: float | None = None,
    ) -> RasterPatch:
        """Extract a square patch centred on ``(lon, lat)``.

        Args:
            path: Raster path (through the Dataset Manager).
            lon / lat: Query point in WGS-84.
            size: Patch edge (default from config).
            pad_mode: ``"constant"`` or ``"reflect"``.
            pad_value: Fill for constant padding.

        Returns:
            A :class:`RasterPatch` (array, mask, bounds, provenance).

        Raises:
            PatchOutOfBoundsError: When the patch cannot be produced.
        """
        edge = size or self.default_size
        mode = pad_mode or self.default_pad_mode
        value = self.default_pad_value if pad_value is None else pad_value

        record = self.metadata_source.image_metadata(path)
        transform, width, height = _transform_from_record(record)

        row, col = geographic_to_raster_index(transform, record.crs, lon, lat)
        requested = patch_window(transform, row, col, edge)
        full_bounds = window_bounds(transform, requested)

        # Clamp the window to the raster extent.
        clamped = _clamp_window(requested, width, height)
        if clamped is None:
            if self.edge_correction:
                # The raster does not intersect the requested window at all.
                raise PatchOutOfBoundsError(
                    f"Patch is entirely outside the raster: {path}",
                    detail={"lon": lon, "lat": lat, "window": _window_tuple(requested)},
                )
            raise PatchOutOfBoundsError(
                f"Patch out of bounds and edge_correction disabled: {path}"
            )

        row_off, col_off, win_h, win_w = _window_tuple(clamped)
        data = self.reader.read_window(path, (row_off, col_off, win_h, win_w), band=1)
        data = np.asarray(data, dtype="float32")

        # Pad to the requested size.
        padded = (win_h, win_w) != (edge, edge)
        if padded:
            data, mask = _pad_to_size(data, edge, mode=mode, value=value)
        else:
            mask = np.ones(data.shape, dtype=bool)

        return RasterPatch(
            path=path,
            array=data,
            mask=mask,
            requested_size=edge,
            window=_window_tuple(clamped),
            bounds=full_bounds,
            crs=record.crs,
            resolution=tuple(record.pixel_size) if record.pixel_size else (0.0, 0.0),
            center_lon=lon,
            center_lat=lat,
            padded=padded,
            extra={
                "index_type": record.index_type,
                "resolution": record.resolution,
                "observation_date": record.observation_date.isoformat()
                if record.observation_date
                else None,
            },
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _transform_from_record(record: ImageRecordRef) -> tuple[Affine, int, int]:
    """Rebuild a north-up affine from a metadata record (bounds + pixel size).

    Assumes north-up orientation (a == 0 and e == 0), which holds for
    Sentinel-2 tiles.
    """
    if not record.pixel_size or not record.bounds:
        raise PatchOutOfBoundsError(
            "Raster metadata lacks pixel_size/bounds required for patch math",
            detail=record.path,
        )
    left, _bottom, right, top = record.bounds
    xres, yres = record.pixel_size
    if xres <= 0 or yres <= 0:
        raise PatchOutOfBoundsError(
            f"Non-positive pixel size: {record.pixel_size}", detail=record.path
        )
    expected_width = (right - left) / xres
    if abs(expected_width - record.width) > 0.5:  # type: ignore[operator]
        raise PatchOutOfBoundsError(
            "Raster is not north-up or bounds/pixel_size are inconsistent",
            detail=record.path,
        )
    transform = Affine(xres, 0.0, left, 0.0, -yres, top)
    return transform, int(record.width or 0), int(record.height or 0)


def _clamp_window(
    window: Any, width: int, height: int
) -> Any | None:
    """Intersect a rasterio Window with the raster extent; None if empty."""
    from rasterio.windows import Window

    row_off = max(0, int(window.row_off))
    col_off = max(0, int(window.col_off))
    row_stop = min(height, int(window.row_off + window.height))
    col_stop = min(width, int(window.col_off + window.width))
    if row_off >= row_stop or col_off >= col_stop:
        return None
    return Window(col_off, row_off, col_stop - col_off, row_stop - row_off)


def _window_tuple(window: Any) -> tuple[int, int, int, int]:
    """Normalise a Window to ``(row_off, col_off, height, width)``."""
    return int(window.row_off), int(window.col_off), int(window.height), int(window.width)


def _pad_to_size(
    data: np.ndarray, size: int, *, mode: str, value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a smaller array to ``(size, size)`` and produce a validity mask."""
    height, width = data.shape
    mask = np.zeros((size, size), dtype=bool)
    mask[:height, :width] = True

    out = np.full((size, size), value, dtype="float32")
    out[:height, :width] = data

    if mode == "reflect":
        try:
            out = np.pad(data, ((0, size - height), (0, size - width)), mode="reflect")
        except Exception:  # noqa: BLE001 - reflect fails near corners
            out = np.full((size, size), value, dtype="float32")
            out[:height, :width] = data
    return out, mask
