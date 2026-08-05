"""Small internal SQLite helper shared by the metadata store, registry and cache.

SQLite is used for three purposes in the Dataset Manager:

* the metadata store,
* the dataset registry + version history,
* the scan/inventory cache.

The three backends share the same connection discipline, so that logic lives
here instead of being duplicated. Every public function opens a fresh
connection per call, which keeps the module thread-safe (SQLite connections
are not shareable across threads) at negligible cost for the low write
volumes involved. WAL journaling avoids read/write lock contention.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .exceptions import DatasetManagerError

#: DDL for the dataset version-history table. Shared by the registry (which
#: owns the database file and cleans up history on removal) and the version
#: manager (which writes snapshots). Kept in one place to avoid duplication.
VERSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL,
    version     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    message     TEXT NOT NULL DEFAULT '',
    checksum    TEXT,
    file_count  INTEGER NOT NULL DEFAULT 0,
    root_path   TEXT,
    is_current  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(dataset_id, version)
);
CREATE INDEX IF NOT EXISTS idx_versions_dataset ON versions (dataset_id);
"""


def connect(db_path: str | Path, *, schema: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with sane production pragmas.

    Args:
        db_path: Database file path. Parent directories are created.
        schema: Optional ``CREATE TABLE ...`` statements executed on connect.

    Returns:
        A configured :class:`sqlite3.Connection` (dict rows, WAL journal).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(path), timeout=30)
    except sqlite3.Error as exc:
        raise DatasetManagerError(
            f"Could not open SQLite database: {exc}", detail=str(path)
        ) from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    if schema:
        conn.executescript(schema)
        conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Context manager that commits on success and rolls back on error."""
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def execute(
    conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> sqlite3.Cursor:
    """Execute a single statement and return the cursor."""
    return conn.execute(sql, params)


def executemany(
    conn: sqlite3.Connection, sql: str, rows: Sequence[Sequence[Any]]
) -> None:
    """Execute a statement against many parameter rows."""
    conn.executemany(sql, rows)


def query(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """Run a SELECT and return a list of plain dicts."""
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def query_one(
    conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> dict[str, Any] | None:
    """Run a SELECT and return the first row as a dict (or None)."""
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row is not None else None
