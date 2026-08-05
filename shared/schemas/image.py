"""Raster / image metadata schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..enums import IndexType, Resolution


@dataclass(slots=True)
class RasterMetadataSchema:
    """Header-only metadata of a GeoTIFF (never loads the raster data)."""

    path: Path
    filename: str
    index_type: IndexType = IndexType.NONE
    resolution: Resolution = Resolution.UNKNOWN
    year: int | None = None
    observation_date: str | None = None
    width: int = 0
    height: int = 0
    dtype: str = "unknown"
    bands: int = 0
    crs: str | None = None
    pixel_size: tuple[float, float] | None = None
    bounds: tuple[float, float, float, float] | None = None
    file_size: int = 0
    sha256: str | None = None
    driver: str | None = None
    compression: str | None = None
    crs_source: str = "rasterio"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "filename": self.filename,
            "index_type": self.index_type.value,
            "resolution": self.resolution.value,
            "year": self.year,
            "observation_date": self.observation_date,
            "width": self.width,
            "height": self.height,
            "dtype": self.dtype,
            "bands": self.bands,
            "crs": self.crs,
            "pixel_size": list(self.pixel_size) if self.pixel_size else None,
            "bounds": list(self.bounds) if self.bounds else None,
            "file_size": self.file_size,
            "sha256": self.sha256,
            "driver": self.driver,
            "compression": self.compression,
            "crs_source": self.crs_source,
        }


@dataclass(slots=True)
class ImageDatasetLocationSchema:
    """Location of a single image record inside a catalog."""

    path: Path
    index_type: IndexType = IndexType.NONE
    resolution: Resolution = Resolution.UNKNOWN
    year: int | None = None
    observation_date: str | None = None
    relative_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "index_type": self.index_type.value,
            "resolution": self.resolution.value,
            "year": self.year,
            "observation_date": self.observation_date,
            "relative_path": self.relative_path,
        }


@dataclass(slots=True)
class ImageDatasetRecordSchema:
    """A catalog record for one image product."""

    handle: str
    path: Path
    source: str = "kaggle"
    fetched_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "path": str(self.path),
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "metadata": self.metadata,
        }
