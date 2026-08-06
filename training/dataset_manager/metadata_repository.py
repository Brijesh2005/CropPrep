"""Extended metadata persistence — provider / spatial / temporal / patch tables.

The R1.2 metadata store (`SQLiteMetadataStore`) holds one row per file in the
``metadata_records`` table. R2.2 extends the **same** ``metadata.db`` with four
more tables so every layer of the Dataset Manager is auditable:

* ``provider_metadata`` — registered providers, their status and capabilities.
* ``spatial_metadata``   — named locations (villages / districts) + coordinates.
* ``temporal_metadata``  — index x year x resolution availability counts.
* ``patch_metadata``     — every patch extraction request / result.

The repository shares the SQLite connection discipline from
:mod:`training.dataset_manager._db` (one short-lived connection per call,
WAL journaling).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ._db import connect, executemany, execute, query, query_one
from .models import PatchMetadata, SpatialRecord, TemporalRecord
from .providers.models import ProviderRegistration

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_metadata (
    name         TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'not_initialized',
    available    INTEGER NOT NULL DEFAULT 0,
    priority     INTEGER NOT NULL DEFAULT 100,
    features     TEXT,
    details      TEXT,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spatial_metadata (
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    latitude     REAL NOT NULL,
    longitude    REAL NOT NULL,
    district     TEXT,
    metadata_json TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (name, kind)
);
CREATE INDEX IF NOT EXISTS idx_spatial_kind   ON spatial_metadata (kind);
CREATE INDEX IF NOT EXISTS idx_spatial_district ON spatial_metadata (district);

CREATE TABLE IF NOT EXISTS temporal_metadata (
    index_type         TEXT NOT NULL,
    year               INTEGER NOT NULL,
    resolution         TEXT NOT NULL,
    count              INTEGER NOT NULL DEFAULT 0,
    observation_months TEXT,
    observation_dates  TEXT,
    updated_at         TEXT NOT NULL,
    PRIMARY KEY (index_type, year, resolution)
);

CREATE TABLE IF NOT EXISTS patch_metadata (
    patch_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL,
    center_x    REAL NOT NULL,
    center_y    REAL NOT NULL,
    size        INTEGER NOT NULL,
    band        INTEGER NOT NULL DEFAULT 1,
    crs         TEXT,
    resolution  TEXT NOT NULL DEFAULT 'UNKNOWN',
    padded      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
"""


class MetadataRepository:
    """Persistence for the extended Dataset Manager metadata tables.

    Args:
        db_path: Path to the (shared) ``metadata.db``. The four extra tables
            are created idempotently alongside ``metadata_records``.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        connect(self.db_path, schema=_SCHEMA)

    # -- Provider metadata ----------------------------------------------------- #

    def save_provider(
        self,
        registration: ProviderRegistration,
        *,
        status: str | None = None,
        available: bool | None = None,
    ) -> None:
        if status is None:
            try:
                status = getattr(registration.provider, "status", "unknown") or "unknown"
            except Exception:  # noqa: BLE001 - best-effort
                status = "unknown"
        if available is None:
            try:
                available = bool(registration.provider.available())
            except Exception:  # noqa: BLE001 - best-effort
                available = False
        try:
            manifest = getattr(registration.provider, "manifest", lambda: None)()
        except Exception:  # noqa: BLE001 - best-effort
            manifest = None
        with connect(self.db_path) as conn:
            execute(
                conn,
                """
                INSERT INTO provider_metadata (
                    name, kind, status, available, priority, features, details, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    kind = excluded.kind,
                    status = excluded.status,
                    available = excluded.available,
                    priority = excluded.priority,
                    features = excluded.features,
                    details = excluded.details,
                    updated_at = excluded.updated_at
                """,
                (
                    registration.name,
                    registration.kind,
                    status,
                    1 if available else 0,
                    registration.priority,
                    json.dumps(_capability_features(registration)),
                    _json_or_none(manifest),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def list_providers(self) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = query(
                conn,
                "SELECT * FROM provider_metadata ORDER BY priority DESC, name",
            )
        return [_decode_details(r) for r in rows]

    def provider_count(self) -> int:
        with connect(self.db_path) as conn:
            row = query_one(conn, "SELECT COUNT(*) AS n FROM provider_metadata")
            return int(row["n"]) if row else 0

    # -- Spatial metadata ------------------------------------------------------ #

    def save_spatial(self, record: SpatialRecord) -> None:
        with connect(self.db_path) as conn:
            execute(
                conn,
                """
                INSERT INTO spatial_metadata (
                    name, kind, latitude, longitude, district, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, kind) DO UPDATE SET
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    district = excluded.district,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.name,
                    record.kind,
                    record.latitude,
                    record.longitude,
                    record.district,
                    json.dumps(record.metadata, ensure_ascii=False, default=str),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def save_spatial_many(self, records: list[SpatialRecord]) -> int:
        if not records:
            return 0
        rows = [
            (
                r.name, r.kind, r.latitude, r.longitude, r.district,
                json.dumps(r.metadata, ensure_ascii=False, default=str),
                datetime.now().isoformat(),
            )
            for r in records
        ]
        with connect(self.db_path) as conn:
            executemany(
                conn,
                """
                INSERT INTO spatial_metadata (
                    name, kind, latitude, longitude, district, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, kind) DO UPDATE SET
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    district = excluded.district,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            conn.commit()
        return len(records)

    def list_spatial(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is None:
            sql = "SELECT * FROM spatial_metadata ORDER BY kind, name"
            params: tuple[Any, ...] = ()
        else:
            sql = "SELECT * FROM spatial_metadata WHERE kind = ? ORDER BY name"
            params = (kind,)
        with connect(self.db_path) as conn:
            rows = query(conn, sql, params)
        return [_decode_spatial(r) for r in rows]

    def spatial_count(self) -> int:
        with connect(self.db_path) as conn:
            row = query_one(conn, "SELECT COUNT(*) AS n FROM spatial_metadata")
            return int(row["n"]) if row else 0

    # -- Temporal metadata ----------------------------------------------------- #

    def save_temporal(self, record: TemporalRecord) -> None:
        with connect(self.db_path) as conn:
            execute(
                conn,
                """
                INSERT INTO temporal_metadata (
                    index_type, year, resolution, count, observation_months,
                    observation_dates, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_type, year, resolution) DO UPDATE SET
                    count = excluded.count,
                    observation_months = excluded.observation_months,
                    observation_dates = excluded.observation_dates,
                    updated_at = excluded.updated_at
                """,
                (
                    record.index_type,
                    record.year,
                    record.resolution,
                    record.count,
                    json.dumps(sorted(record.observation_months)),
                    json.dumps(record.observation_dates, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def list_temporal(
        self,
        *,
        index_type: str | None = None,
        year: int | None = None,
        resolution: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if index_type:
            where.append("index_type = ?")
            params.append(index_type)
        if year is not None:
            where.append("year = ?")
            params.append(int(year))
        if resolution:
            where.append("resolution = ?")
            params.append(resolution)
        sql = "SELECT * FROM temporal_metadata"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY year, index_type, resolution"
        with connect(self.db_path) as conn:
            rows = query(conn, sql, params)
        return [_decode_temporal(r) for r in rows]

    # -- Patch metadata -------------------------------------------------------- #

    def save_patch(self, metadata: PatchMetadata) -> int:
        with connect(self.db_path) as conn:
            cursor = execute(
                conn,
                """
                INSERT INTO patch_metadata (
                    path, center_x, center_y, size, band, crs, resolution, padded, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(metadata.path),
                    metadata.center[0],
                    metadata.center[1],
                    metadata.size,
                    metadata.band,
                    metadata.crs,
                    metadata.resolution,
                    1 if metadata.padded else 0,
                    metadata.created_at.isoformat(),
                ),
            )
            conn.commit()
        return int(cursor.lastrowid)

    def list_patches(self, limit: int = 100) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = query(
                conn,
                "SELECT * FROM patch_metadata ORDER BY patch_id DESC LIMIT ?",
                (int(limit),),
            )
        return [_decode_patch(r) for r in rows]

    def patch_count(self) -> int:
        with connect(self.db_path) as conn:
            row = query_one(conn, "SELECT COUNT(*) AS n FROM patch_metadata")
            return int(row["n"]) if row else 0


# --------------------------------------------------------------------------- #
# Row decoding helpers
# --------------------------------------------------------------------------- #


def _capability_features(registration: ProviderRegistration) -> list[str]:
    try:
        caps = registration.provider.capabilities()
        return list(getattr(caps, "features", []) or [])
    except Exception:  # noqa: BLE001 - best effort
        return []


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    data = value.to_dict() if hasattr(value, "to_dict") else value
    return json.dumps(data, ensure_ascii=False, default=str)


def _decode_details(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["available"] = bool(row["available"])
    for key in ("features", "details"):
        if out.get(key):
            try:
                out[key] = json.loads(out[key])
            except (ValueError, TypeError):
                pass
    return out


def _decode_spatial(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["metadata"] = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
    out.pop("metadata_json", None)
    return out


def _decode_temporal(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["observation_months"] = (
        json.loads(row["observation_months"]) if row.get("observation_months") else []
    )
    out["observation_dates"] = (
        json.loads(row["observation_dates"]) if row.get("observation_dates") else []
    )
    return out


def _decode_patch(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["padded"] = bool(row["padded"])
    out["center"] = (row["center_x"], row["center_y"])
    out.pop("center_x", None)
    out.pop("center_y", None)
    return out
