"""Coordinate / CRS helpers for STAM.

Everything here works on rasterio :class:`Affine` transforms and pyproj
CRS/Transformer objects. Points are handled in WGS-84 (lon, lat) at the STAM
boundary and transformed into a raster's own CRS before any pixel math — this
keeps patch extraction axis-aligned and avoids costly image reprojection in
the common case.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pyproj
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol, xy
from rasterio.windows import Window

from .exceptions import CRSMismatchError

WGS84 = CRS.from_epsg(4326)


def normalise_crs(crs: Any) -> CRS | None:
    """Coerce a CRS into a :class:`rasterio.crs.CRS` (or None).

    Accepts ``None``, ``"EPSG:4326"`` / ``"4326"`` strings, integers,
    pyproj CRS and rasterio CRS objects.
    """
    if crs is None:
        return None
    if isinstance(crs, CRS):
        return crs
    # A bare numeric string like "4326" is an EPSG code.
    if isinstance(crs, str) and crs.strip().isdigit():
        crs = int(crs.strip())
    try:
        return CRS.from_user_input(crs)
    except Exception:  # noqa: BLE001 - invalid CRS input
        return None


def crs_to_epsg(crs: Any) -> int | None:
    """Return the EPSG code of a CRS, or None when not defined/known."""
    normalised = normalise_crs(crs)
    if normalised is None:
        return None
    try:
        return normalised.to_epsg()
    except Exception:  # noqa: BLE001
        return None


def validate_crs(crs: Any) -> bool:
    """True when ``crs`` is a usable (parseable) CRS definition."""
    return normalise_crs(crs) is not None


def assert_same_crs(a: Any, b: Any, *, context: str = "") -> CRS:
    """Ensure two CRS definitions are equivalent; raise otherwise.

    Returns the normalised CRS on success.

    Raises:
        CRSMismatchError: When either CRS is missing or they differ.
    """
    ca = normalise_crs(a)
    cb = normalise_crs(b)
    if ca is None or cb is None:
        raise CRSMismatchError(
            f"CRS comparison needs both sides defined: {a!r} vs {b!r}",
            detail=context,
        )
    if ca != cb:
        raise CRSMismatchError(
            f"CRS mismatch: {a} vs {b}",
            detail=context,
        )
    return ca


def make_transformer(crs_from: Any, crs_to: Any) -> pyproj.Transformer:
    """Build an always-xy transformer between two CRS definitions."""
    return pyproj.Transformer.from_crs(
        normalise_crs(crs_from) or WGS84,
        normalise_crs(crs_to) or WGS84,
        always_xy=True,
    )


def transform_point(crs_from: Any, crs_to: Any, lon: float, lat: float) -> tuple[float, float]:
    """Transform a single ``(lon, lat)`` point into ``crs_to`` coordinates."""
    x, y = make_transformer(crs_from, crs_to).transform(lon, lat)
    return float(x), float(y)


def transform_points(
    crs_from: Any, crs_to: Any, lons: Iterable[float], lats: Iterable[float]
) -> tuple[list[float], list[float]]:
    """Transform lists of lon/lat into the target CRS (vectors)."""
    xs = np.asarray(list(lons), dtype="float64")
    ys = np.asarray(list(lats), dtype="float64")
    tx, ty = make_transformer(crs_from, crs_to).transform(xs, ys)
    return list(tx), list(ty)


def world_to_pixel(transform: Affine, x: float, y: float) -> tuple[int, int]:
    """Map world ``(x, y)`` to raster ``(row, col)``."""
    rows, cols = rowcol(transform, x, y)
    return int(rows), int(cols)


def pixel_to_world(transform: Affine, row: int, col: int) -> tuple[float, float]:
    """Map raster ``(row, col)`` to the centre world ``(x, y)``."""
    x, y = xy(transform, row, col, offset="center")
    return float(x), float(y)


def geographic_to_raster_index(
    transform: Affine,
    raster_crs: Any,
    lon: float,
    lat: float,
) -> tuple[int, int]:
    """Project ``(lon, lat)`` into the raster's grid and return ``(row, col)``.

    The point is first transformed from WGS-84 to the raster CRS, then
    inverted through the affine transform.
    """
    x, y = transform_point(WGS84, raster_crs or WGS84, lon, lat)
    return world_to_pixel(transform, x, y)


def patch_window(transform: Affine, row: int, col: int, size: int) -> Window:
    """A centred ``size x size`` window around a pixel ``(row, col)``."""
    half = size // 2
    row_start = row - half
    col_start = col - half
    return Window(col_start, row_start, size, size)


def window_affine(transform: Affine, window: Window) -> Affine:
    """The affine transform of a sub-window (in global CRS coordinates)."""
    return Affine(
        transform.a,
        transform.b,
        transform.c + window.col_off * transform.a,
        transform.d,
        transform.e,
        transform.f + window.row_off * transform.e,
    )


def window_bounds(transform: Affine, window: Window) -> tuple[float, float, float, float]:
    """``(left, bottom, right, top)`` of a window in CRS units."""
    left, top = window_affine(transform, window) * (0, 0)
    right, bottom = window_affine(transform, window) * (window.width, window.height)
    return left, bottom, right, top


def raster_bounds_from_metadata(
    transform: Affine, width: int, height: int
) -> tuple[float, float, float, float]:
    """``(left, bottom, right, top)`` of a whole raster."""
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return left, bottom, right, top
