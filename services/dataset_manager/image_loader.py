"""Lazy GeoTIFF loader: header metadata, previews and windowed reads.

The loader is the **only** module that reads raster (GeoTIFF) files. It
supports the NDVI / EVI indices and the R10m / R20m resolution bands by
classifying them from the file path / name (see the scanner) and attaches
that classification to every :class:`RasterMetadata` result.

Design notes:

* **Never loads full rasters implicitly.** :meth:`read_metadata` touches only
  the header; :meth:`read_window` reads a bounded window; :meth:`load` is the
  single explicit full-read escape hatch (documented as memory heavy).
* **Two backends.** When GDAL/rasterio is available it is used (full CRS and
  compression support). A self-contained TIFF/IFD parser
  (:func:`_light_tiff_metadata`) provides header-only metadata without GDAL —
  used as a fallback and by tests that force it.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import CorruptedDatasetError, UnsupportedFormatError
from .interfaces import ImageLoader
from .logger import get_logger
from .models import IndexType, Resolution, RasterMetadata
from .utils import (
    classify_index_type_from_path,
    classify_resolution_from_path,
    extract_year_from_path,
    is_geotiff_bytes,
    parse_observation_date,
)

logger = get_logger("image_loader")

_RASTER_SUFFIXES = {".tif", ".tiff"}
_GDAL_COMPRESSION = {
    1: "none", 5: "lzw", 6: "jpeg", 7: "jpeg", 8: "deflate",
    32946: "deflate", 32773: "packbits", 50000: "zstd", 50001: "webp",
}
_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


class RasterioImageLoader(ImageLoader):
    """Concrete :class:`ImageLoader` backed by rasterio (GDAL)."""

    def __init__(self) -> None:
        self._rasterio = None
        try:
            import rasterio  # type: ignore[import-not-found]

            self._rasterio = rasterio
        except ImportError:  # pragma: no cover - fallback path exercised in tests
            self._rasterio = None

    # -- Public API ------------------------------------------------------------ #

    def is_supported(self, path: Path) -> bool:
        """True when the path has a raster extension (magic check optional)."""
        return Path(path).suffix.lower() in _RASTER_SUFFIXES

    def read_metadata(self, path: Path, *, prefer_light: bool = False) -> RasterMetadata:
        """Read header-only metadata of a GeoTIFF without loading data.

        Args:
            path: Raster file.
            prefer_light: Force the pure-python TIFF/IFD parser (used in
                tests and when GDAL is unavailable).

        Returns:
            A populated :class:`RasterMetadata`.

        Raises:
            UnsupportedFormatError: For non-raster files.
            CorruptedDatasetError: When the TIFF header cannot be parsed.
        """
        path = Path(path)
        if not self.is_supported(path):
            raise UnsupportedFormatError(f"Not a raster file: {path}", detail=str(path))
        if not is_geotiff_bytes(path):
            raise CorruptedDatasetError(
                f"File is not a valid TIFF (bad magic bytes): {path}", detail=str(path)
            )

        common = self._classify(path)

        if not prefer_light and self._rasterio is not None:
            try:
                return self._metadata_via_rasterio(path, common)
            except CorruptedDatasetError:
                # Fall through to the light parser; if that also fails the
                # light parser raises with full context.
                pass

        return self._metadata_via_light(path, common)

    def preview(self, path: Path) -> dict[str, Any]:
        """Dimensions, dtype and a small sampled statistics block.

        The statistics come from a bounded 10x10 corner window — the full
        raster is never loaded.
        """
        metadata = self.read_metadata(path)
        data: dict[str, Any] = metadata.to_dict()
        data["sample_stats"] = {}
        if self._rasterio is not None and metadata.width and metadata.height:
            try:
                rasterio = self._rasterio
                height = min(10, metadata.height)
                width = min(10, metadata.width)
                with rasterio.open(path) as src:
                    window = rasterio.windows.Window(0, 0, width, height)
                    sample = src.read(
                        1,
                        window=window,
                        out_shape=(height, width),
                        resampling=rasterio.enums.Resampling.nearest,
                    )
                sample = sample.astype("float64")
                finite = sample[np.isfinite(sample)]
                if finite.size:
                    data["sample_stats"] = {
                        "min": float(finite.min()),
                        "max": float(finite.max()),
                        "mean": float(finite.mean()),
                        "std": float(finite.std()),
                    }
            except Exception as exc:  # noqa: BLE001 - preview is best effort
                logger.debug("Preview statistics unavailable", extra={"path": str(path), "reason": str(exc)})
        return data

    def read_window(
        self,
        path: Path,
        *,
        window: tuple[int, int, int, int],
        band: int = 1,
    ) -> np.ndarray:
        """Read a bounded window of a band.

        Args:
            path: Raster file.
            window: ``(row_offset, col_offset, height, width)``.
            band: 1-based band index (default 1).

        Returns:
            A 2-D :class:`numpy.ndarray` for the requested window.
        """
        if self._rasterio is None:
            raise UnsupportedFormatError(
                "Windowed reads require rasterio (GDAL)", detail="rasterio"
            )
        rasterio = self._rasterio
        row_off, col_off, height, width = window
        with rasterio.open(path) as src:
            rwin = rasterio.windows.Window(col_off, row_off, width, height)
            return src.read(band, window=rwin)

    def load(self, path: Path, band: int = 1) -> np.ndarray:
        """Explicitly load a full band into memory (memory heavy).

        Prefer :meth:`read_window` or :meth:`preview` for production use.
        """
        if self._rasterio is None:
            raise UnsupportedFormatError(
                "Full reads require rasterio (GDAL)", detail="rasterio"
            )
        rasterio = self._rasterio
        with rasterio.open(path) as src:
            return src.read(band)

    # -- Internals ------------------------------------------------------------- #

    def _classify(self, path: Path) -> dict[str, Any]:
        return {
            "index_type": classify_index_type_from_path(path),
            "resolution": classify_resolution_from_path(path),
            "year": extract_year_from_path(path),
            "observation_date": parse_observation_date(path),
            "file_size": path.stat().st_size,
        }

    def _metadata_via_rasterio(self, path: Path, common: dict[str, Any]) -> RasterMetadata:
        rasterio = self._rasterio
        try:
            with rasterio.open(path) as src:
                crs = src.crs.to_string() if src.crs is not None else None
                bounds = src.bounds  # BoundingBox(left, bottom, right, top)
                return RasterMetadata(
                    path=path,
                    filename=path.name,
                    index_type=common["index_type"],
                    resolution=common["resolution"],
                    year=common["year"],
                    observation_date=common["observation_date"],
                    width=src.width,
                    height=src.height,
                    dtype=src.dtypes[0] if src.dtypes else "unknown",
                    bands=src.count,
                    crs=crs,
                    pixel_size=(float(src.res[0]), float(src.res[1])),
                    bounds=(float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)),
                    file_size=common["file_size"],
                    driver=src.driver,
                    compression=src.compression,
                    crs_source="rasterio",
                )
        except Exception as exc:  # noqa: BLE001
            raise CorruptedDatasetError(
                f"rasterio could not read raster header: {path.name}",
                detail=str(exc),
            ) from exc

    def _metadata_via_light(self, path: Path, common: dict[str, Any]) -> RasterMetadata:
        info = _light_tiff_metadata(path)
        return RasterMetadata(
            path=path,
            filename=path.name,
            index_type=common["index_type"],
            resolution=common["resolution"],
            year=common["year"],
            observation_date=common["observation_date"],
            width=info["width"],
            height=info["height"],
            dtype=info["dtype"],
            bands=info["bands"],
            crs=None,
            pixel_size=info["pixel_size"],
            bounds=info["bounds"],
            file_size=common["file_size"],
            driver="GTiff",
            compression=info["compression"],
            crs_source="light",
        )


# --------------------------------------------------------------------------- #
# Lightweight TIFF / IFD parser (no GDAL required)
# --------------------------------------------------------------------------- #


def _light_tiff_metadata(path: str | Path) -> dict[str, Any]:
    """Parse a TIFF header using only ``struct``.

    Reads the byte order, magic, IFD and a focused subset of tags:
    width (256), height (257), bits per sample (258), compression (259),
    samples per pixel (277), sample format (339), pixel scale (33550) and
    tiepoint (33922). No pixel data is ever read.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                raise CorruptedDatasetError(f"Truncated TIFF header: {path}")
            order = head[:2]
            if order == b"II":
                endian = "<"
            elif order == b"MM":
                endian = ">"
            else:
                raise CorruptedDatasetError(f"Unknown TIFF byte order: {path}")
            magic = struct.unpack(endian + "H", head[2:4])[0]
            if magic != 42:
                raise CorruptedDatasetError(f"Invalid TIFF magic value: {path}")
            ifd_offset = struct.unpack(endian + "I", head[4:8])[0]
            fh.seek(ifd_offset)
            count = struct.unpack(endian + "H", fh.read(2))[0]
            tags: dict[int, Any] = {}
            for _ in range(count):
                entry = fh.read(12)
                if len(entry) < 12:
                    break
                tag, typ, cnt = struct.unpack(endian + "HHI", entry[:8])
                tags[tag] = _read_tiff_value(fh, endian, typ, cnt, entry[8:12])
    except OSError as exc:
        raise CorruptedDatasetError(
            f"Could not read TIFF file: {path}", detail=str(exc)
        ) from exc

    width = tags.get(256, [0])[0] if tags.get(256) else 0
    height = tags.get(257, [0])[0] if tags.get(257) else 0
    bits = tags.get(258, [8])[0] if tags.get(258) else 8
    sample_format = tags.get(339, [1])[0] if tags.get(339) else 1
    bands = tags.get(277, [1])[0] if tags.get(277) else 1
    compression = _GDAL_COMPRESSION.get(tags.get(259, [1])[0], "unknown")

    pixel_scale = tags.get(33550) or [1.0, 1.0, 0.0]
    xres, yres = float(pixel_scale[0]), float(pixel_scale[1])
    tiepoint = tags.get(33922) or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    origin_x, origin_y = float(tiepoint[3]), float(tiepoint[4])

    bounds = None
    pixel_size = None
    if width > 0 and height > 0:
        pixel_size = (xres, yres)
        bounds = (
            origin_x,
            origin_y - (height * yres),
            origin_x + (width * xres),
            origin_y,
        )

    return {
        "width": width,
        "height": height,
        "bands": bands,
        "dtype": _light_dtype(bits, sample_format),
        "compression": compression,
        "pixel_size": pixel_size,
        "bounds": bounds,
    }


def _read_tiff_value(fh, endian: str, typ: int, count: int, inline: bytes) -> Any:
    """Read a TIFF IFD value, following the offset when it exceeds 4 bytes."""
    size = _TIFF_TYPE_SIZE.get(typ, 0)
    total = size * count
    if total <= 4:
        data = inline[:total]
    else:
        offset = struct.unpack(endian + "I", inline)[0]
        position = fh.tell()
        try:
            fh.seek(offset)
            data = fh.read(total)
        finally:
            fh.seek(position)

    fmt = endian
    if typ == 1:  # BYTE
        return list(data)
    if typ == 2:  # ASCII
        return data.rstrip(b"\0").decode("latin-1")
    if typ == 3:  # SHORT
        return list(struct.unpack(f"{fmt}{count}H", data))
    if typ == 4 or typ == 13:  # LONG / IFD
        return list(struct.unpack(f"{fmt}{count}I", data))
    if typ == 5:  # RATIONAL
        raws = struct.unpack(f"{fmt}{count * 2}I", data)
        return [raws[i] / raws[i + 1] if raws[i + 1] else 0.0 for i in range(0, len(raws), 2)]
    if typ == 11:  # FLOAT
        return list(struct.unpack(f"{fmt}{count}f", data))
    if typ == 12:  # DOUBLE
        return list(struct.unpack(f"{fmt}{count}d", data))
    return data.hex()  # pragma: no cover - unsupported tag types


def _light_dtype(bits: int, sample_format: int) -> str:
    """Map (bits per sample, sample format) to a numpy dtype name."""
    if sample_format == 3:  # IEEE float
        return {32: "float32", 64: "float64"}.get(bits, "float32")
    if sample_format == 2:  # signed integer
        return {8: "int8", 16: "int16", 32: "int32", 64: "int64"}.get(bits, "int16")
    return {8: "uint8", 16: "uint16", 32: "uint32", 64: "uint64"}.get(bits, "uint8")
