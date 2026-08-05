"""Generic dataset metadata schemas (dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..enums import DatasetStatus, FileCategory, IndexType, Resolution


@dataclass(slots=True)
class FileEntrySchema:
    """One file discovered during a scan."""

    path: Path
    relative_path: str
    category: FileCategory
    index_type: IndexType = IndexType.NONE
    resolution: Resolution = Resolution.UNKNOWN
    year: int | None = None
    size_bytes: int = 0
    mtime: float = 0.0
    sha256: str | None = None
    extension: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "category": self.category.value,
            "index_type": self.index_type.value,
            "resolution": self.resolution.value,
            "year": self.year,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "sha256": self.sha256,
            "extension": self.extension,
        }


@dataclass(slots=True)
class DatasetInventorySchema:
    """The complete inventory produced by a scan."""

    root: Path
    entries: list[FileEntrySchema] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=datetime.now)
    duration_s: float = 0.0
    source: str = "scan"

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "total": len(self.entries),
            "csv": 0,
            "geotiff": 0,
            "ndvi": 0,
            "evi": 0,
            "r10m": 0,
            "r20m": 0,
        }
        for entry in self.entries:
            if entry.category is FileCategory.CSV:
                counts["csv"] += 1
            elif entry.category is FileCategory.GEOTIFF:
                counts["geotiff"] += 1
            if entry.index_type is IndexType.NDVI:
                counts["ndvi"] += 1
            elif entry.index_type is IndexType.EVI:
                counts["evi"] += 1
            if entry.resolution is Resolution.R10M:
                counts["r10m"] += 1
            elif entry.resolution is Resolution.R20M:
                counts["r20m"] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "scanned_at": self.scanned_at.isoformat(),
            "duration_s": self.duration_s,
            "source": self.source,
            "counts": self.counts(),
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass(slots=True)
class MetadataRecordSchema:
    """A single row of dataset metadata (one record per file)."""

    path: Path
    relative_path: str
    category: FileCategory
    index_type: IndexType = IndexType.NONE
    resolution: Resolution = Resolution.UNKNOWN
    year: int | None = None
    observation_date: str | None = None
    crs: str | None = None
    file_size: int = 0
    sha256: str | None = None
    row_count: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_path: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "relative_path": self.relative_path,
            "category": self.category.value,
            "index_type": self.index_type.value,
            "resolution": self.resolution.value,
            "year": self.year,
            "observation_date": self.observation_date,
            "crs": self.crs,
            "file_size": self.file_size,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "created_at": self.created_at.isoformat(),
            "extra": self.extra,
        }
        if include_path:
            data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class DatasetSummarySchema:
    """Human + machine readable summary of a dataset."""

    name: str
    root: Path
    status: DatasetStatus = DatasetStatus.PENDING
    version: str = "0.0.0"
    total_files: int = 0
    total_size_bytes: int = 0
    csv_count: int = 0
    geotiff_count: int = 0
    years_covered: list[int] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "status": self.status.value,
            "version": self.version,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "csv_count": self.csv_count,
            "geotiff_count": self.geotiff_count,
            "years_covered": self.years_covered,
            "generated_at": self.generated_at.isoformat(),
        }
