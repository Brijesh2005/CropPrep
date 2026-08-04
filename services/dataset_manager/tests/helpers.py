"""Plain test helpers (not pytest fixtures).

Functions here can be called directly from fixtures and tests without
triggering pytest's fixture wrapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def make_tiff(
    path: Path,
    *,
    width: int = 20,
    height: int = 20,
    dtype: str = "float32",
    crs: str | None = "EPSG:32643",
    seed: int = 7,
    fill: float | None = None,
    driver: str = "GTiff",
    origin: tuple[float, float] = (74.8, 13.0),
) -> Path:
    """Create a small valid GeoTIFF for tests.

    Args:
        origin: Top-left ``(lon, lat)`` of the raster (default near
            Dakshina Kannada). Pixel size is 0.0001 deg (north-up).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if fill is not None:
        array = np.full((height, width), fill, dtype=dtype)
    else:
        array = rng.uniform(0.1, 0.9, (height, width)).astype(dtype)
    with rasterio.open(
        path, "w", driver=driver, height=height, width=width, count=1,
        dtype=dtype, crs=crs,
        transform=from_origin(origin[0], origin[1], 0.0001, 0.0001),
    ) as dst:
        dst.write(array, 1)
    return path
