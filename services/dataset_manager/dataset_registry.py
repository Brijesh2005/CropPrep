"""Dataset registry: lifecycle, paths, status, checksum and provenance.

The registry tracks every managed dataset in a SQLite table:

* dataset versions and their root paths,
* lifecycle status (pending → downloading → downloaded → validating →
  validated → ready / failed),
* content checksum + file count (integrity fingerprint),
* last update timestamp and source (e.g. ``"kaggle"``),
* free-form JSON metadata.

The registry is the source of truth the API/backend will query to answer
"what datasets exist, in what state, at what version" without touching the
filesystem.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ._db import VERSIONS_SCHEMA, connect, execute, query, query_one, transaction
from .exceptions import RegistryError
from .interfaces import Registry
from .logger import get_logger
from .models import DatasetStatus

logger = get_logger("registry")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS registry (
    dataset_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'unknown',
    version       TEXT NOT NULL DEFAULT '0.0.0',
    root_path     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    checksum      TEXT,
    file_count    INTEGER NOT NULL DEFAULT 0,
    last_updated  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_registry_name    ON registry (name);
CREATE INDEX IF NOT EXISTS idx_registry_status  ON registry (status);
"""
# The version-history table shares this database so registry removals can
# clean up history transactionally (see VERSIONS_SCHEMA in ``_db``).
_SCHEMA += VERSIONS_SCHEMA


class SQLiteRegistry(Registry):
    """Concrete :class:`Registry` persisted in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        connect(self.db_path, schema=_SCHEMA)

    # -- Writes ---------------------------------------------------------------- #

    def register(
        self,
        *,
        name: str,
        source: str,
        root_path: Path,
        version: str = "0.0.0",
        status: DatasetStatus = DatasetStatus.PENDING,
    ) -> int:
        now = datetime.now().isoformat()
        with connect(self.db_path) as conn:
            with transaction(conn):
                cursor = execute(
                    conn,
                    """
                    INSERT INTO registry
                        (name, source, version, root_path, status, checksum,
                         file_count, last_updated, created_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, NULL, 0, ?, ?, '{}')
                    """,
                    (name, source, version, str(Path(root_path).expanduser().resolve()),
                     status.value, now, now),
                )
                dataset_id = int(cursor.lastrowid)
        logger.info(
            "Registered dataset",
            extra={"dataset_id": dataset_id, "dataset_name": name, "version": version},
        )
        return dataset_id

    def update_status(self, dataset_id: int, status: DatasetStatus) -> None:
        self._update(dataset_id, status=status.value)

    def update_checksum(self, dataset_id: int, checksum: str, file_count: int) -> None:
        self._update(dataset_id, checksum=checksum, file_count=file_count)

    def set_version(self, dataset_id: int, version: str) -> None:
        self._update(dataset_id, version=version)

    def update_metadata(self, dataset_id: int, metadata: dict[str, Any]) -> None:
        import json as _json

        self._update(dataset_id, metadata_json=_json.dumps(metadata, default=str))

    def remove(self, dataset_id: int) -> bool:
        with connect(self.db_path) as conn:
            with transaction(conn):
                cursor = execute(conn, "DELETE FROM registry WHERE dataset_id = ?", (dataset_id,))
                execute(conn, "DELETE FROM versions WHERE dataset_id = ?", (dataset_id,))
                return cursor.rowcount > 0

    # -- Reads ----------------------------------------------------------------- #

    def get(self, dataset_id: int) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = query_one(conn, "SELECT * FROM registry WHERE dataset_id = ?", (dataset_id,))
        return _normalise(row) if row else None

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = query_one(
                conn,
                "SELECT * FROM registry WHERE name = ? ORDER BY dataset_id DESC LIMIT 1",
                (name,),
            )
        return _normalise(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = query(conn, "SELECT * FROM registry ORDER BY dataset_id DESC")
        return [_normalise(row) for row in rows]

    def list_by_status(self, status: DatasetStatus) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = query(
                conn, "SELECT * FROM registry WHERE status = ? ORDER BY dataset_id DESC",
                (status.value,),
            )
        return [_normalise(row) for row in rows]

    def count(self) -> int:
        with connect(self.db_path) as conn:
            row = query_one(conn, "SELECT COUNT(*) AS n FROM registry")
        return int(row["n"]) if row else 0

    # -- Internals ------------------------------------------------------------- #

    def _update(self, dataset_id: int, **fields: Any) -> None:
        if not fields:
            return
        # Reject unknown columns to catch typos at the boundary.
        allowed = {"status", "checksum", "file_count", "version", "metadata_json"}
        unknown = set(fields) - allowed
        if unknown:
            raise RegistryError(f"Unknown registry fields: {sorted(unknown)}")
        columns = ", ".join(f"{key} = ?" for key in fields)
        # Parameter order must match: SET <fields>, last_updated = ?, WHERE dataset_id = ?
        params = [*fields.values(), datetime.now().isoformat(), dataset_id]
        with connect(self.db_path) as conn:
            with transaction(conn):
                # Keys are allowlisted above; values are bound parameters.
                execute(
                    conn,
                    f"UPDATE registry SET {columns}, last_updated = ? WHERE dataset_id = ?",  # nosec B608
                    params,
                )


def _normalise(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": row["dataset_id"],
        "name": row["name"],
        "source": row["source"],
        "version": row["version"],
        "root_path": row["root_path"],
        "status": row["status"],
        "checksum": row["checksum"],
        "file_count": row["file_count"],
        "last_updated": row["last_updated"],
        "created_at": row["created_at"],
        "metadata_json": row["metadata_json"],
    }
