"""Data models for the Dataset Manager provider layer.

The provider layer wraps the *sources* of data (a Git repository of tabular
CSVs, the Kaggle Sentinel-2 imagery dataset, ...) behind a stable contract so
that :class:`~training.dataset_manager.manager.DatasetManager` never reads
files directly — it only ever talks to providers.

These models are the *lingua franca* of the provider layer:

* :class:`TabularDatasetInfo` / :class:`TabularCatalog` — discovery results of
  the tabular provider.
* :class:`TabularJoinSpec` — a declarative join between two discovered
  datasets.
* :class:`ImageDatasetLocation` — where the imagery dataset lives on disk and
  whether it has been downloaded / materialised.
* :class:`ImageCatalog` — the classified inventory (NDVI / EVI, year,
  resolution) of the imagery dataset.
* :class:`PatchRequest` — a declarative raster patch retrieval request.
* :class:`ProviderStatus` / :class:`ProviderManifest` — introspection used by
  the bootstrap and diagnostics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import CSVProfile, FileEntry

# --------------------------------------------------------------------------- #
# Provider introspection
# --------------------------------------------------------------------------- #


class ProviderStatus(str, enum.Enum):
    """Lifecycle status of a provider."""

    NOT_INITIALIZED = "not_initialized"
    READY = "ready"
    MISSING_DATA = "missing_data"
    ERROR = "error"


@dataclass(slots=True)
class ProviderManifest:
    """Machine readable description of a provider instance."""

    name: str
    kind: str
    status: ProviderStatus
    available: bool
    root: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status.value,
            "available": self.available,
            "root": str(self.root) if self.root is not None else None,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Tabular provider models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TabularDatasetInfo:
    """One discovered tabular (CSV) dataset.

    Attributes:
        name: Stable dataset name (file stem). Discovery is automatic and
            name-based — no filenames are ever hardcoded by consumers.
        path: Absolute path of the CSV file.
        relative_path: Path relative to the provider root (POSIX style).
        size_bytes: Byte size on disk.
        profile: Optional schema / quality profile (computed on demand).
    """

    name: str
    path: Path
    relative_path: str
    size_bytes: int = 0
    profile: CSVProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "profile": self.profile.to_dict() if self.profile is not None else None,
        }


@dataclass(slots=True)
class TabularCatalog:
    """Result of a tabular discovery pass."""

    root: Path
    datasets: list[TabularDatasetInfo] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)

    def names(self) -> list[str]:
        """Stable dataset names, sorted."""
        return sorted(info.name for info in self.datasets)

    def by_name(self, name: str) -> TabularDatasetInfo | None:
        for info in self.datasets:
            if info.name == name:
                return info
        return None

    def total_size(self) -> int:
        return sum(info.size_bytes for info in self.datasets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "discovered_at": self.discovered_at.isoformat(),
            "count": len(self.datasets),
            "total_size_bytes": self.total_size(),
            "datasets": [info.to_dict() for info in self.datasets],
        }


@dataclass(slots=True)
class TabularJoinSpec:
    """Declarative join between two discovered tabular datasets.

    Attributes:
        name: Left dataset name.
        other: Right dataset name.
        on: Join key column(s) present in both datasets.
        how: ``inner`` | ``left`` | ``right`` | ``outer``.
        suffixes: Column suffix disambiguation (default ``_left``/``_right``).
    """

    name: str
    other: str
    on: str | list[str]
    how: str = "inner"
    suffixes: tuple[str, str] = ("_left", "_right")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "other": self.other,
            "on": self.on,
            "how": self.how,
            "suffixes": list(self.suffixes),
        }


# --------------------------------------------------------------------------- #
# Image provider models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ImageDatasetLocation:
    """Where the imagery dataset lives and how far materialisation got."""

    handle: str
    root: Path
    downloaded: bool = False
    materialized: bool = False
    version: str | None = None
    files: int = 0
    size_bytes: int = 0
    cache_root: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "root": str(self.root),
            "downloaded": self.downloaded,
            "materialized": self.materialized,
            "version": self.version,
            "files": self.files,
            "size_bytes": self.size_bytes,
            "cache_root": str(self.cache_root) if self.cache_root else None,
        }


@dataclass(slots=True)
class ImageCatalog:
    """Classified inventory of the imagery dataset.

    Attributes:
        location: The on-disk location the catalog was built from.
        entries: All scanned file entries.
        ndvi: Entries classified as NDVI rasters.
        evi: Entries classified as EVI rasters.
        years: Distinct years present, sorted.
        resolutions: Distinct resolution bands present, sorted.
        counts: Flat count summary (total / csv / geotiff / ndvi / evi / ...).
    """

    location: ImageDatasetLocation
    entries: list[FileEntry] = field(default_factory=list)
    ndvi: list[FileEntry] = field(default_factory=list)
    evi: list[FileEntry] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    resolutions: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "ndvi_count": len(self.ndvi),
            "evi_count": len(self.evi),
            "years": self.years,
            "resolutions": self.resolutions,
            "counts": self.counts,
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass(slots=True)
class PatchRequest:
    """A declarative raster patch retrieval request.

    Attributes:
        path: Raster file to read.
        center: Geographic center ``(x, y)`` in the raster's CRS units
            (longitude, latitude for EPSG:4326 products).
        size: Square patch edge length in pixels.
        band: 1-based band index (default 1).
    """

    path: Path
    center: tuple[float, float]
    size: int
    band: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "center": list(self.center),
            "size": self.size,
            "band": self.band,
        }
