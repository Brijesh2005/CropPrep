"""Shared data models (dataclasses + enums) for the Dataset Manager.

These types are the *lingua franca* between modules: the scanner produces
:class:`DatasetInventory` / :class:`FileEntry`, the validator consumes them
and emits :class:`ValidationReport` / :class:`ValidationIssue`, the metadata
generator produces :class:`MetadataRecord`, and the registry / version
manager use :class:`RegistryEntry` / :class:`VersionEntry`.

All records are plain dataclasses so they serialize to JSON trivially
(:meth:`~object.__dict__` compatible, plus explicit ``to_dict`` helpers for
enums and paths).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from shared.enums import (
    DatasetStatus,
    FAILING_SEVERITY,
    FileCategory,
    IndexType,
    Resolution,
    Severity,
)


# --------------------------------------------------------------------------- #
# Scan models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FileEntry:
    """One file discovered during a scan.

    Attributes:
        path: Absolute path to the file.
        relative_path: POSIX-style path relative to the scan root.
        category: File category (csv / geotiff / other).
        index_type: Vegetation index detected for rasters (NDVI/EVI/NONE).
        resolution: Spatial resolution band (R10m/R20m/UNKNOWN).
        year: Calendar year extracted from the path, when detectable.
        size_bytes: File size on disk.
        mtime: Modification time (epoch seconds).
        sha256: Content hash, present only when hashing was requested.
        extension: Lower-cased file extension without the leading dot.
    """

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileEntry":
        return cls(
            path=Path(data["path"]),
            relative_path=data["relative_path"],
            category=FileCategory(data["category"]),
            index_type=IndexType(data["index_type"]),
            resolution=Resolution(data["resolution"]),
            year=data.get("year"),
            size_bytes=data.get("size_bytes", 0),
            mtime=data.get("mtime", 0.0),
            sha256=data.get("sha256"),
            extension=data.get("extension", ""),
        )


@dataclass(slots=True)
class DatasetInventory:
    """The complete inventory produced by a scan.

    Attributes:
        root: Scan root directory.
        entries: All discovered file entries.
        scanned_at: Time the scan was performed.
        duration_s: Wall-clock duration of the scan in seconds.
        source: Where the inventory came from — ``"scan"`` or ``"cache"``.
    """

    root: Path
    entries: list[FileEntry] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=datetime.now)
    duration_s: float = 0.0
    source: str = "scan"

    # -- Convenience accessors ------------------------------------------------- #
    def by_category(self) -> dict[FileCategory, list[FileEntry]]:
        out: dict[FileCategory, list[FileEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.category, []).append(entry)
        return out

    def by_index(self) -> dict[IndexType, list[FileEntry]]:
        out: dict[IndexType, list[FileEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.index_type, []).append(entry)
        return out

    def by_resolution(self) -> dict[Resolution, list[FileEntry]]:
        out: dict[Resolution, list[FileEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.resolution, []).append(entry)
        return out

    def by_year(self) -> dict[int, list[FileEntry]]:
        out: dict[int, list[FileEntry]] = {}
        for entry in self.entries:
            if entry.year is not None:
                out.setdefault(entry.year, []).append(entry)
        return out

    def csv_files(self) -> list[FileEntry]:
        return [e for e in self.entries if e.category is FileCategory.CSV]

    def geotiff_files(self) -> list[FileEntry]:
        return [e for e in self.entries if e.category is FileCategory.GEOTIFF]

    def total_size(self) -> int:
        return sum(e.size_bytes for e in self.entries)

    def counts(self) -> dict[str, int]:
        """Summary counts keyed by a flat, JSON friendly label."""
        counts: dict[str, int] = {
            "total": len(self.entries),
            "csv": len(self.csv_files()),
            "geotiff": len(self.geotiff_files()),
            "ndvi": 0,
            "evi": 0,
            "r10m": 0,
            "r20m": 0,
        }
        for entry in self.entries:
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetInventory":
        return cls(
            root=Path(data["root"]),
            entries=[FileEntry.from_dict(e) for e in data.get("entries", [])],
            scanned_at=datetime.fromisoformat(data["scanned_at"]),
            duration_s=data.get("duration_s", 0.0),
            source=data.get("source", "scan"),
        )


# --------------------------------------------------------------------------- #
# CSV models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CSVProfile:
    """Inferred schema and quality profile of a single CSV file."""

    path: Path
    filename: str
    encoding: str = "utf-8"
    row_count: int | None = None
    column_count: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    total_missing: int = 0
    size_bytes: int = 0
    has_header: bool = True
    #: Optional extra profiling output (e.g. numeric ``statistics``).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "filename": self.filename,
            "encoding": self.encoding,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "missing_values": self.missing_values,
            "total_missing": self.total_missing,
            "size_bytes": self.size_bytes,
            "has_header": self.has_header,
            "extra": self.extra,
        }


# --------------------------------------------------------------------------- #
# Raster / GeoTIFF models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RasterMetadata:
    """Header-only metadata of a GeoTIFF (never loads the raster data).

    Attributes:
        path: Absolute path to the raster.
        filename: Basename of the raster.
        index_type: Vegetation index (NDVI / EVI / NONE).
        resolution: Resolution band (R10m / R20m / UNKNOWN).
        year: Year extracted from the path, when detectable.
        observation_date: Observation date parsed from the filename, if any.
        width / height: Raster dimensions in pixels.
        dtype: Numpy data type string of the raster bands.
        bands: Number of bands.
        crs: Human readable CRS (e.g. ``"EPSG:32643"``); None when unknown.
        pixel_size: ``(x_res, y_res)`` in map units.
        bounds: ``(left, bottom, right, top)`` in CRS units.
        file_size: Byte size on disk.
        sha256: Content hash when computed.
        driver: GDAL driver name (e.g. ``"GTiff"``).
        compression: Compression scheme (e.g. ``"deflate"``).
        crs_source: How CRS was resolved — ``"rasterio"`` (GDAL geokeys) or
            ``"light"`` (bare IFD pixel scale — CRS stays None).
    """

    path: Path
    filename: str
    index_type: IndexType = IndexType.NONE
    resolution: Resolution = Resolution.UNKNOWN
    year: int | None = None
    observation_date: date | None = None
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
            "observation_date": self.observation_date.isoformat()
            if self.observation_date
            else None,
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


# --------------------------------------------------------------------------- #
# Validation models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ValidationIssue:
    """A single issue found during validation."""

    severity: Severity
    code: str
    category: str
    message: str
    path: str | None = None
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ValidationReport:
    """Aggregated outcome of a validation run."""

    root: Path
    passed: bool
    issues: list[ValidationIssue]
    files_scanned: int
    validated_at: datetime = field(default_factory=datetime.now)

    def by_severity(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for issue in self.issues:
            key = issue.severity.value
            out[key] = out.get(key, 0) + 1
        return out

    def by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for issue in self.issues:
            out[issue.category] = out.get(issue.category, 0) + 1
        return out

    def failing_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity in FAILING_SEVERITY]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "passed": self.passed,
            "files_scanned": self.files_scanned,
            "validated_at": self.validated_at.isoformat(),
            "by_severity": self.by_severity(),
            "by_category": self.by_category(),
            "issues": [i.to_dict() for i in self.issues],
        }

    def write_json(self, out: str | Path) -> Path:
        """Persist the report as pretty JSON and return the written path."""
        import json

        out_path = Path(out)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out_path


# --------------------------------------------------------------------------- #
# Metadata models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MetadataRecord:
    """A single row of dataset metadata (one record per file).

    This is the payload persisted in the SQLite metadata store (and exported
    to Parquet for analytics).
    """

    path: Path
    relative_path: str
    category: FileCategory
    index_type: IndexType = IndexType.NONE
    resolution: Resolution = Resolution.UNKNOWN
    year: int | None = None
    observation_date: date | None = None
    width: int | None = None
    height: int | None = None
    dtype: str | None = None
    bands: int | None = None
    crs: str | None = None
    pixel_size: tuple[float, float] | None = None
    bounds: tuple[float, float, float, float] | None = None
    file_size: int = 0
    sha256: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    columns_json: str | None = None
    encoding: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_path: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "relative_path": self.relative_path,
            "category": self.category.value,
            "index_type": self.index_type.value,
            "resolution": self.resolution.value,
            "year": self.year,
            "observation_date": self.observation_date.isoformat()
            if self.observation_date
            else None,
            "width": self.width,
            "height": self.height,
            "dtype": self.dtype,
            "bands": self.bands,
            "crs": self.crs,
            "pixel_size": list(self.pixel_size) if self.pixel_size else None,
            "bounds": list(self.bounds) if self.bounds else None,
            "file_size": self.file_size,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns_json": self.columns_json,
            "encoding": self.encoding,
            "created_at": self.created_at.isoformat(),
            "extra": self.extra,
        }
        if include_path:
            data["path"] = str(self.path)
        return data


# --------------------------------------------------------------------------- #
# Historical context model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HistoricalContext:
    """Temporal availability of image records for a recurring season window.

    Produced by :meth:`DatasetManager.get_historical_context`: for a location
    and a season (expressed as calendar months), it reports which years in the
    catalog contain satellite records in that same season — the multi-year
    "same location + same season" context gathered before model inference.
    """

    #: Calendar months the season occupies (e.g. ``[6, 7, 8, 9, 10]`` for
    #: Kharif, ``[11, 12, 1, 2, 3]`` for Rabi). None when no window was set.
    window_months: list[int] | None = None
    #: Years in the catalog that contain records in the season window.
    years: list[int] = field(default_factory=list)
    #: Per-year record counts, e.g. ``{2020: {"records": 6, "ndvi": 3, "evi": 3}}``.
    per_year: dict[int, dict[str, int]] = field(default_factory=dict)
    total_records: int = 0
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_months": self.window_months,
            "years": self.years,
            "per_year": {str(k): v for k, v in self.per_year.items()},
            "total_records": self.total_records,
            "generated_at": self.generated_at.isoformat(),
        }


# --------------------------------------------------------------------------- #
# Registry / version models
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RegistryEntry:
    """A dataset row in the registry."""

    dataset_id: int
    name: str
    source: str
    version: str
    root_path: Path
    status: DatasetStatus
    checksum: str | None = None
    file_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    metadata_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "version": self.version,
            "root_path": str(self.root_path),
            "status": self.status.value,
            "checksum": self.checksum,
            "file_count": self.file_count,
            "last_updated": self.last_updated.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata_json": self.metadata_json,
        }


@dataclass(slots=True)
class VersionEntry:
    """A single version in a dataset's version history."""

    dataset_id: int
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    checksum: str | None = None
    file_count: int = 0
    root_path: Path | None = None
    is_current: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "message": self.message,
            "checksum": self.checksum,
            "file_count": self.file_count,
            "root_path": str(self.root_path) if self.root_path else None,
            "is_current": self.is_current,
        }


# --------------------------------------------------------------------------- #
# Summary model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DatasetSummary:
    """Human + machine readable summary of a dataset inventory."""

    root: Path
    total_files: int
    total_size_bytes: int
    csv_count: int
    geotiff_count: int
    other_count: int
    ndvi_count: int
    evi_count: int
    files_by_year: dict[int, int]
    files_by_resolution: dict[str, int]
    years_covered: list[int]
    index_types_present: list[str]
    resolutions_present: list[str]
    csv_row_estimate: int | None = None
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "csv_count": self.csv_count,
            "geotiff_count": self.geotiff_count,
            "other_count": self.other_count,
            "ndvi_count": self.ndvi_count,
            "evi_count": self.evi_count,
            "files_by_year": {str(k): v for k, v in sorted(self.files_by_year.items())},
            "files_by_resolution": self.files_by_resolution,
            "years_covered": self.years_covered,
            "index_types_present": self.index_types_present,
            "resolutions_present": self.resolutions_present,
            "csv_row_estimate": self.csv_row_estimate,
            "generated_at": self.generated_at.isoformat(),
        }
