"""Metadata generation and persistence.

Two components live here:

* :class:`MetadataGenerator` — builds one :class:`MetadataRecord` per scanned
  file: year, observation date, resolution, image type (NDVI/EVI), bounding
  box, file size, CRS, pixel size, band count, hash, creation time and more.
* :class:`SQLiteMetadataStore` — persists and queries those records.

**Why SQLite (over Parquet)?** — The metadata store is written *incrementally*
(dataset grows file by file), queried per-file at inference time (STAM
lookups), and shared between the scanner and the validator. SQLite gives
transactional upserts, unique-path indexing, point lookups and zero extra
dependencies, all of which Parquet (a write-once, full-rewrite columnar
format) does not provide for this access pattern. Parquet remains valuable
for *analytics*, which is why :meth:`SQLiteMetadataStore.export_parquet`
provides a one-command export of the entire table for research workloads.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ._db import connect, execute, query, query_one
from .config import MetadataConfig
from .exceptions import InvalidMetadataError
from .interfaces import CSVLoader, ImageLoader, MetadataGenerator, MetadataStore
from .logger import get_logger
from .models import (
    DatasetInventory,
    FileCategory,
    FileEntry,
    IndexType,
    MetadataRecord,
    Resolution,
    RasterMetadata,
)
from .utils import count_lines_fast, run_parallel, sha256_file

logger = get_logger("metadata")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata_records (
    relative_path    TEXT PRIMARY KEY,
    path             TEXT NOT NULL,
    category         TEXT NOT NULL,
    index_type       TEXT NOT NULL DEFAULT 'NONE',
    resolution       TEXT NOT NULL DEFAULT 'UNKNOWN',
    year             INTEGER,
    observation_date TEXT,
    width            INTEGER,
    height           INTEGER,
    dtype            TEXT,
    bands            INTEGER,
    crs              TEXT,
    pixel_size       TEXT,
    bounds           TEXT,
    file_size        INTEGER NOT NULL DEFAULT 0,
    sha256           TEXT,
    row_count        INTEGER,
    column_count     INTEGER,
    columns_json     TEXT,
    encoding         TEXT,
    created_at       TEXT NOT NULL,
    extra            TEXT
);
CREATE INDEX IF NOT EXISTS idx_meta_year       ON metadata_records (year);
CREATE INDEX IF NOT EXISTS idx_meta_category   ON metadata_records (category);
CREATE INDEX IF NOT EXISTS idx_meta_index      ON metadata_records (index_type);
CREATE INDEX IF NOT EXISTS idx_meta_resolution ON metadata_records (resolution);
"""


class MetadataGeneratorImpl(MetadataGenerator):
    """Concrete :class:`MetadataGenerator`.

    Args:
        config: Metadata configuration section.
        csv_loader: :class:`CSVLoader` used to profile tabular files.
        image_loader: :class:`ImageLoader` used to read raster headers.
        store: :class:`MetadataStore` receiving the generated records.
    """

    def __init__(
        self,
        config: MetadataConfig | None = None,
        *,
        csv_loader: CSVLoader,
        image_loader: ImageLoader,
        store: MetadataStore,
    ) -> None:
        self.config = config or MetadataConfig()
        self.csv_loader = csv_loader
        self.image_loader = image_loader
        self.store = store

    def generate(
        self, root: Path, inventory: DatasetInventory, *, force: bool = False
    ) -> list[MetadataRecord]:
        """Generate metadata for every file in ``inventory``.

        Args:
            root: Dataset root (used to build absolute paths).
            inventory: Scanner inventory.
            force: Regenerate records that already exist and match on size.

        Returns:
            The list of records generated (persisted via the store).
        """
        records: list[MetadataRecord] = []
        pending: list[FileEntry] = []

        for entry in inventory.entries:
            if entry.category is FileCategory.OTHER:
                continue
            if not force:
                existing = self.store.get(entry.path)
                if existing is not None and existing.file_size == entry.size_bytes:
                    continue  # already indexed and unchanged
            pending.append(entry)

        logger.info(
            "Generating metadata",
            extra={"pending": len(pending), "total": len(inventory.entries), "force": force},
        )

        build = self._build_record
        results = run_parallel(pending, build, workers=self.config.workers)
        for result in results:
            if isinstance(result, MetadataRecord):
                records.append(result)
            else:  # pragma: no cover - defensive; _build_record does not raise
                logger.error("Metadata build failed", extra={"error": str(result)})

        if records:
            self.store.save_many(records)
        return records

    # -- Internals ------------------------------------------------------------- #

    def _build_record(self, entry: FileEntry) -> MetadataRecord:
        if entry.category is FileCategory.CSV:
            return self._build_csv_record(entry)
        if entry.category is FileCategory.GEOTIFF:
            return self._build_raster_record(entry)
        raise InvalidMetadataError(
            f"Unsupported category for metadata: {entry.category}", detail=str(entry.path)
        )

    def _build_csv_record(self, entry: FileEntry) -> MetadataRecord:
        try:
            columns = list(self.csv_loader.preview(entry.path, n_rows=0).columns)
            encoding = self.csv_loader.guess_encoding(entry.path)
        except Exception as exc:  # noqa: BLE001
            raise InvalidMetadataError(
                f"Could not profile CSV for metadata: {entry.relative_path}",
                detail=str(exc),
            ) from exc

        row_count = None
        if entry.size_bytes > 0:
            row_count = max(0, count_lines_fast(entry.path) - 1)

        return MetadataRecord(
            path=entry.path,
            relative_path=entry.relative_path,
            category=entry.category,
            index_type=IndexType.NONE,
            resolution=Resolution.UNKNOWN,
            year=entry.year,
            observation_date=None,
            file_size=entry.size_bytes,
            sha256=sha256_file(entry.path) if self.config.compute_hashes else None,
            row_count=row_count,
            column_count=len(columns),
            columns_json=json.dumps(columns, ensure_ascii=False),
            encoding=encoding,
        )

    def _build_raster_record(self, entry: FileEntry) -> MetadataRecord:
        try:
            meta: RasterMetadata = self.image_loader.read_metadata(entry.path)
        except Exception as exc:  # noqa: BLE001
            raise InvalidMetadataError(
                f"Could not read raster metadata: {entry.relative_path}",
                detail=str(exc),
            ) from exc
        return MetadataRecord(
            path=entry.path,
            relative_path=entry.relative_path,
            category=entry.category,
            index_type=meta.index_type,
            resolution=meta.resolution,
            year=meta.year,
            observation_date=meta.observation_date,
            width=meta.width,
            height=meta.height,
            dtype=meta.dtype,
            bands=meta.bands,
            crs=meta.crs,
            pixel_size=meta.pixel_size,
            bounds=meta.bounds,
            file_size=entry.size_bytes,
            sha256=sha256_file(entry.path) if self.config.compute_hashes else None,
        )


class SQLiteMetadataStore(MetadataStore):
    """Concrete :class:`MetadataStore` persisted in SQLite (WAL mode)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        connect(self.db_path, schema=_SCHEMA)

    # -- Writes ---------------------------------------------------------------- #

    def save(self, record: MetadataRecord) -> None:
        with connect(self.db_path) as conn:
            execute(conn, _UPSERT_SQL, _record_params(record))
            conn.commit()

    def save_many(self, records: list[MetadataRecord]) -> int:
        with connect(self.db_path) as conn:
            conn.executemany(_UPSERT_SQL, [_record_params(r) for r in records])
            conn.commit()
        return len(records)

    # -- Reads ----------------------------------------------------------------- #

    def get(self, path: Path) -> MetadataRecord | None:
        key = str(path)
        with connect(self.db_path) as conn:
            row = query_one(
                conn,
                "SELECT * FROM metadata_records WHERE relative_path = ? OR path = ? LIMIT 1",
                (key, key),
            )
        return _row_to_record(row) if row else None

    def query(self, **filters: Any) -> list[MetadataRecord]:
        where: list[str] = []
        params: list[Any] = []
        for column, value in filters.items():
            if value is None:
                continue
            if column == "year":
                where.append("year = ?")
                params.append(int(value))
            elif column == "index_type":
                where.append("index_type = ?")
                params.append(value.value if isinstance(value, IndexType) else str(value).upper())
            elif column == "resolution":
                where.append("resolution = ?")
                params.append(value.value if isinstance(value, Resolution) else str(value))
            elif column == "category":
                where.append("category = ?")
                params.append(value.value if isinstance(value, FileCategory) else str(value).lower())
            else:
                raise InvalidMetadataError(
                    f"Unsupported metadata query filter: {column}"
                )
        sql = "SELECT * FROM metadata_records"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with connect(self.db_path) as conn:
            rows = query(conn, sql, params)
        return [_row_to_record(row) for row in rows]

    def all(self) -> list[MetadataRecord]:
        with connect(self.db_path) as conn:
            rows = query(conn, "SELECT * FROM metadata_records")
        return [_row_to_record(row) for row in rows]

    def count(self) -> int:
        with connect(self.db_path) as conn:
            row = query_one(conn, "SELECT COUNT(*) AS n FROM metadata_records")
        return int(row["n"]) if row else 0

    def close(self) -> None:
        # Connections are short-lived (one per call); nothing to release.
        return None

    # -- Export ---------------------------------------------------------------- #

    def export_parquet(self, path: Path) -> Path:
        """Export the full metadata table to a Parquet file.

        Requires ``pyarrow`` (or ``fastparquet``). Raises
        :class:`InvalidMetadataError` when neither engine is installed.
        """
        records = self.all()
        rows = [r.to_dict(include_path=True) for r in records]
        frame = pd.DataFrame.from_records(rows)
        # Parquet cannot represent python dict/list cells (e.g. ``extra``);
        # serialise any structured object column to a JSON string first.
        for column in frame.columns:
            if frame[column].dtype == object and frame[column].map(
                lambda v: isinstance(v, (dict, list))
            ).any():
                frame[column] = frame[column].map(
                    lambda v: json.dumps(v, ensure_ascii=False, default=str)
                    if isinstance(v, (dict, list))
                    else v
                )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(out, index=False)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise InvalidMetadataError(
                "Parquet export requires pyarrow or fastparquet; install with "
                "`pip install pyarrow`",
                detail=str(exc),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise InvalidMetadataError(
                f"Parquet export failed: {exc}", detail=str(out)
            ) from exc
        logger.info("Exported metadata to Parquet", extra={"path": str(out), "rows": len(rows)})
        return out


# --------------------------------------------------------------------------- #
# Row <-> record conversion helpers
# --------------------------------------------------------------------------- #

_UPSERT_SQL = """
INSERT INTO metadata_records (
    relative_path, path, category, index_type, resolution, year,
    observation_date, width, height, dtype, bands, crs, pixel_size, bounds,
    file_size, sha256, row_count, column_count, columns_json, encoding,
    created_at, extra
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(relative_path) DO UPDATE SET
    path = excluded.path,
    category = excluded.category,
    index_type = excluded.index_type,
    resolution = excluded.resolution,
    year = excluded.year,
    observation_date = excluded.observation_date,
    width = excluded.width,
    height = excluded.height,
    dtype = excluded.dtype,
    bands = excluded.bands,
    crs = excluded.crs,
    pixel_size = excluded.pixel_size,
    bounds = excluded.bounds,
    file_size = excluded.file_size,
    sha256 = excluded.sha256,
    row_count = excluded.row_count,
    column_count = excluded.column_count,
    columns_json = excluded.columns_json,
    encoding = excluded.encoding,
    created_at = excluded.created_at,
    extra = excluded.extra
"""


def _record_params(record: MetadataRecord) -> tuple[Any, ...]:
    return (
        record.relative_path,
        str(record.path),
        record.category.value,
        record.index_type.value,
        record.resolution.value,
        record.year,
        record.observation_date.isoformat() if record.observation_date else None,
        record.width,
        record.height,
        record.dtype,
        record.bands,
        record.crs,
        json.dumps(list(record.pixel_size)) if record.pixel_size else None,
        json.dumps(list(record.bounds)) if record.bounds else None,
        record.file_size,
        record.sha256,
        record.row_count,
        record.column_count,
        record.columns_json,
        record.encoding,
        record.created_at.isoformat(),
        json.dumps(record.extra, ensure_ascii=False, default=str),
    )


def _row_to_record(row: dict[str, Any]) -> MetadataRecord:
    def _parse_float_pair(raw: str | None) -> tuple[float, float] | None:
        if not raw:
            return None
        try:
            values = json.loads(raw)
            return (float(values[0]), float(values[1]))
        except (ValueError, TypeError, IndexError):
            return None

    def _parse_quad(raw: str | None) -> tuple[float, float, float, float] | None:
        if not raw:
            return None
        try:
            values = json.loads(raw)
            return tuple(float(v) for v in values)  # type: ignore[return-value]
        except (ValueError, TypeError, IndexError):
            return None

    obs_date = None
    if row.get("observation_date"):
        try:
            obs_date = datetime.fromisoformat(row["observation_date"]).date()
        except ValueError:
            obs_date = None

    return MetadataRecord(
        path=Path(row["path"]),
        relative_path=row["relative_path"],
        category=FileCategory(row["category"]),
        index_type=IndexType(row["index_type"] or "NONE"),
        resolution=Resolution(row["resolution"] or "UNKNOWN"),
        year=row.get("year"),
        observation_date=obs_date,
        width=row.get("width"),
        height=row.get("height"),
        dtype=row.get("dtype"),
        bands=row.get("bands"),
        crs=row.get("crs"),
        pixel_size=_parse_float_pair(row.get("pixel_size")),
        bounds=_parse_quad(row.get("bounds")),
        file_size=row.get("file_size") or 0,
        sha256=row.get("sha256"),
        row_count=row.get("row_count"),
        column_count=row.get("column_count"),
        columns_json=row.get("columns_json"),
        encoding=row.get("encoding"),
        created_at=datetime.fromisoformat(row["created_at"]),
        extra=json.loads(row["extra"]) if row.get("extra") else {},
    )
