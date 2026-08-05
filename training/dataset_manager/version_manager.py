"""Semantic versioning of datasets.

Versioning follows MAJOR.MINOR.PATCH:

* **MAJOR** — incompatible structural change (schema / layout breaking).
* **MINOR** — backward compatible additions (new years, new index types).
* **PATCH** — corrections (fixes, metadata refresh).

Every bump records a :class:`VersionEntry` snapshot (version, timestamp,
message, checksum, file count, root path, ``is_current``) and updates the
dataset's current version in the registry. The version table lives in the
same SQLite database as the registry so they can be queried together.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ._db import VERSIONS_SCHEMA, connect, execute, query, query_one, transaction
from .exceptions import RegistryError
from .interfaces import Registry, VersionManager
from .logger import get_logger
from .models import VersionEntry

logger = get_logger("versioning")

_SCHEMA = VERSIONS_SCHEMA

_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class SQLiteVersionManager(VersionManager):
    """Concrete :class:`VersionManager` persisted alongside the registry.

    Args:
        db_path: Database file (the same one used by the registry).
        registry: :class:`Registry` used to read/write the current version.
    """

    def __init__(self, db_path: str | Path, registry: Registry) -> None:
        self.db_path = Path(db_path)
        self.registry = registry
        connect(self.db_path, schema=_SCHEMA)

    # -- Reads ----------------------------------------------------------------- #

    def current(self, dataset_id: int) -> str | None:
        with connect(self.db_path) as conn:
            row = query_one(
                conn,
                "SELECT version FROM versions WHERE dataset_id = ? AND is_current = 1",
                (dataset_id,),
            )
        if row is not None:
            return row["version"]
        entry = self.registry.get(dataset_id)
        return entry["version"] if entry else None

    def list(self, dataset_id: int) -> list[VersionEntry]:
        with connect(self.db_path) as conn:
            rows = query(
                conn,
                "SELECT * FROM versions WHERE dataset_id = ? ORDER BY version_id DESC",
                (dataset_id,),
            )
        return [_row_to_entry(row) for row in rows]

    # -- Writes ---------------------------------------------------------------- #

    def snapshot(
        self,
        dataset_id: int,
        version: str,
        *,
        message: str,
        checksum: str | None,
        file_count: int,
    ) -> VersionEntry:
        """Record a version snapshot and mark it current.

        Args:
            dataset_id: Registered dataset id.
            version: Valid semver string.
            message: Snapshot message (why this version).
            checksum: Integrity checksum of the dataset root.
            file_count: Number of files in the snapshot.
        """
        self._validate_version(version)
        if self.registry.get(dataset_id) is None:
            raise RegistryError(f"Dataset not registered: {dataset_id}")

        created = datetime.now().isoformat()
        with connect(self.db_path) as conn:
            with transaction(conn):
                execute(conn, "UPDATE versions SET is_current = 0 WHERE dataset_id = ?", (dataset_id,))
                # Idempotent snapshot: reuse an existing row for this version.
                existing = query_one(
                    conn,
                    "SELECT version_id FROM versions WHERE dataset_id = ? AND version = ?",
                    (dataset_id, version),
                )
                if existing:
                    execute(
                        conn,
                        """
                        UPDATE versions SET message = ?, checksum = ?, file_count = ?,
                               is_current = 1, root_path = ?
                        WHERE version_id = ?
                        """,
                        (message, checksum, file_count, None, existing["version_id"]),
                    )
                else:
                    execute(
                        conn,
                        """
                        INSERT INTO versions
                            (dataset_id, version, created_at, message, checksum,
                             file_count, root_path, is_current)
                        VALUES (?, ?, ?, ?, ?, ?, NULL, 1)
                        """,
                        (dataset_id, version, created, message, checksum, file_count),
                    )
        self.registry.set_version(dataset_id, version)
        if checksum is not None:
            self.registry.update_checksum(dataset_id, checksum, file_count)

        entry = VersionEntry(
            dataset_id=dataset_id,
            version=version,
            created_at=datetime.fromisoformat(created),
            message=message,
            checksum=checksum,
            file_count=file_count,
            is_current=True,
        )
        logger.info(
            "Version snapshot recorded",
            extra={"dataset_id": dataset_id, "version": version, "reason": message},
        )
        return entry

    def bump(
        self, dataset_id: int, part: str = "patch", *, message: str = ""
    ) -> VersionEntry:
        """Bump the current version by one ``part`` (major/minor/patch)."""
        current = self.current(dataset_id) or "0.0.0"
        next_version = bump_version(current, part)
        entry = self.registry.get(dataset_id)
        file_count = entry["file_count"] if entry else 0
        return self.snapshot(
            dataset_id,
            next_version,
            message=message or f"bump {part}",
            checksum=entry["checksum"] if entry else None,
            file_count=file_count,
        )

    def rollback(self, dataset_id: int, version: str) -> VersionEntry:
        """Restore ``version`` as the current version (must exist)."""
        self._validate_version(version)
        with connect(self.db_path) as conn:
            with transaction(conn):
                row = query_one(
                    conn,
                    "SELECT * FROM versions WHERE dataset_id = ? AND version = ?",
                    (dataset_id, version),
                )
                if row is None:
                    raise RegistryError(
                        f"Version not found for rollback: {version}",
                        detail={"dataset_id": dataset_id},
                    )
                execute(conn, "UPDATE versions SET is_current = 0 WHERE dataset_id = ?", (dataset_id,))
                execute(
                    conn,
                    "UPDATE versions SET is_current = 1 WHERE version_id = ?",
                    (row["version_id"],),
                )
        self.registry.set_version(dataset_id, version)
        if row["checksum"] is not None:
            self.registry.update_checksum(dataset_id, row["checksum"], row["file_count"])
        entry = _row_to_entry(row)
        entry.is_current = True
        logger.info(
            "Rolled back dataset version",
            extra={"dataset_id": dataset_id, "version": version},
        )
        return entry

    # -- Internals ------------------------------------------------------------- #

    def _validate_version(self, version: str) -> None:
        if not _VERSION_RE.match(version):
            raise RegistryError(
                f"Invalid semantic version (expected MAJOR.MINOR.PATCH): {version}"
            )


def bump_version(version: str, part: str) -> str:
    """Return the next semver string for ``version`` after bumping ``part``.

    Args:
        version: Current version string (``MAJOR.MINOR.PATCH``).
        part: ``"major"``, ``"minor"`` or ``"patch"``.

    Returns:
        The bumped version string.
    """
    if part not in {"major", "minor", "patch"}:
        raise RegistryError(
            f"Invalid bump part (expected major/minor/patch): {part}"
        )
    try:
        major, minor, patch = (int(x) for x in version.split(".")[:3])
    except ValueError as exc:
        raise RegistryError(f"Invalid version string: {version}") from exc

    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _row_to_entry(row: dict[str, Any]) -> VersionEntry:
    return VersionEntry(
        dataset_id=row["dataset_id"],
        version=row["version"],
        created_at=datetime.fromisoformat(row["created_at"]),
        message=row["message"],
        checksum=row["checksum"],
        file_count=row["file_count"],
        root_path=Path(row["root_path"]) if row.get("root_path") else None,
        is_current=bool(row["is_current"]),
    )
