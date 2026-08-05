"""Disk-backed key/value cache with TTL and prefix invalidation.

Used to avoid repeatedly scanning and re-profiling datasets. Entries are
stored in a SQLite database (WAL mode) with an optional expiry timestamp:

* :meth:`get` returns ``None`` on a miss or when the entry has expired
  (expired entries are lazily pruned).
* :meth:`set` accepts an explicit TTL and falls back to the configured
  default.
* :meth:`delete_prefix` supports grouped invalidation (e.g. all ``scan:``
  keys) — the mechanism the scanner uses to drop stale inventories.

The cache is safe for concurrent readers/writers because each operation opens
its own short-lived connection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ._db import connect, execute, query, query_one, transaction
from .config import CacheConfig
from .exceptions import CacheError
from .interfaces import Cache
from .logger import get_logger

logger = get_logger("cache")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL              -- NULL => never expires
);
CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries (expires_at);
"""


class CacheManager(Cache):
    """Concrete :class:`Cache` implementation persisted in SQLite.

    Args:
        db_path: Database file. When ``enabled`` is False a no-op in-memory
            cache is used (useful for tests and read-only flows).
        config: Cache configuration section.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        config: CacheConfig | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.config = config or CacheConfig()
        # A file-backed cache requires an explicit database path. Without one
        # the cache runs in-process (dict), which is the right default for
        # tests and short-lived flows.
        use_file = db_path is not None and str(db_path) != ":memory:"
        if enabled is None:
            enabled = use_file and self.config.enabled
        self.config = self.config.model_copy(update={"enabled": enabled})
        self.db_path = Path(db_path) if use_file else Path(":memory:")
        self._memory: dict[str, tuple[Any, float | None]] = {}

        if self.config.enabled and use_file:
            try:
                connect(self.db_path, schema=_SCHEMA)
            except Exception as exc:  # noqa: BLE001
                raise CacheError(
                    f"Could not initialise cache: {exc}", detail=str(self.db_path)
                ) from exc

    # -- Public API ------------------------------------------------------------ #

    def get(self, key: str) -> Any | None:
        """Return the deserialised value for ``key``, or None on miss/expiry."""
        if not self.config.enabled:
            return self._memory_get(key)
        try:
            with connect(self.db_path) as conn:
                row = query_one(
                    conn,
                    "SELECT value, expires_at FROM cache_entries WHERE key = ?",
                    (key,),
                )
            if row is None:
                return None
            if row["expires_at"] is not None and row["expires_at"] < time.time():
                self.delete(key)
                return None
            return json.loads(row["value"])
        except Exception as exc:  # noqa: BLE001
            raise CacheError(f"Cache read failed for {key}: {exc}", detail=str(exc)) from exc

    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL."""
        ttl = self.config.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at: float | None = None
        if ttl is not None and ttl >= 0:
            expires_at = time.time() + ttl

        if not self.config.enabled:
            self._memory_set(key, value, expires_at)
            return

        try:
            with connect(self.db_path) as conn:
                with transaction(conn):
                    execute(
                        conn,
                        """
                        INSERT INTO cache_entries (key, value, created_at, expires_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value = excluded.value,
                            created_at = excluded.created_at,
                            expires_at = excluded.expires_at
                        """,
                        (key, json.dumps(value, ensure_ascii=False, default=str),
                         time.time(), expires_at),
                    )
                    self._enforce_capacity(conn)
        except Exception as exc:  # noqa: BLE001
            raise CacheError(f"Cache write failed for {key}: {exc}", detail=str(exc)) from exc

    def delete(self, key: str) -> bool:
        """Remove ``key``; returns True when it existed."""
        if not self.config.enabled:
            return self._memory_delete(key)
        try:
            with connect(self.db_path) as conn:
                with transaction(conn):
                    cursor = execute(conn, "DELETE FROM cache_entries WHERE key = ?", (key,))
                    return cursor.rowcount > 0
        except Exception as exc:  # noqa: BLE001
            raise CacheError(f"Cache delete failed for {key}: {exc}", detail=str(exc)) from exc

    def delete_prefix(self, prefix: str) -> int:
        """Remove all keys starting with ``prefix``; returns the count removed."""
        if not self.config.enabled:
            return self._memory_delete_prefix(prefix)
        try:
            with connect(self.db_path) as conn:
                with transaction(conn):
                    cursor = execute(
                        conn, "DELETE FROM cache_entries WHERE key LIKE ?", (f"{prefix}%",)
                    )
                    return cursor.rowcount
        except Exception as exc:  # noqa: BLE001
            raise CacheError(
                f"Cache prefix delete failed for {prefix}: {exc}", detail=str(exc)
            ) from exc

    def clear(self) -> int:
        """Drop all entries; returns the count removed."""
        if not self.config.enabled:
            count = len(self._memory)
            self._memory.clear()
            return count
        try:
            with connect(self.db_path) as conn:
                with transaction(conn):
                    cursor = execute(conn, "DELETE FROM cache_entries")
                    return cursor.rowcount
        except Exception as exc:  # noqa: BLE001
            raise CacheError(f"Cache clear failed: {exc}", detail=str(exc)) from exc

    def prune(self) -> int:
        """Remove expired entries; returns the count removed."""
        if not self.config.enabled:
            now = time.time()
            expired = [k for k, (_, exp) in self._memory.items() if exp is not None and exp < now]
            for key in expired:
                del self._memory[key]
            return len(expired)
        try:
            with connect(self.db_path) as conn:
                with transaction(conn):
                    cursor = execute(
                        conn, "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                        (time.time(),),
                    )
                    return cursor.rowcount
        except Exception as exc:  # noqa: BLE001
            raise CacheError(f"Cache prune failed: {exc}", detail=str(exc)) from exc

    def size(self) -> int:
        """Number of live entries in the cache."""
        if not self.config.enabled:
            return len(self._memory)
        with connect(self.db_path) as conn:
            row = query_one(conn, "SELECT COUNT(*) AS n FROM cache_entries")
            return int(row["n"]) if row else 0

    # -- Internals ------------------------------------------------------------- #

    def _enforce_capacity(self, conn) -> None:
        """Evict oldest entries beyond ``max_entries`` (best-effort)."""
        if self.config.max_entries <= 0:
            return
        row = query_one(conn, "SELECT COUNT(*) AS n FROM cache_entries")
        count = int(row["n"]) if row else 0
        if count > self.config.max_entries:
            overflow = count - self.config.max_entries
            execute(
                conn,
                """
                DELETE FROM cache_entries
                WHERE key IN (
                    SELECT key FROM cache_entries ORDER BY created_at ASC LIMIT ?
                )
                """,
                (overflow,),
            )

    def _memory_get(self, key: str) -> Any | None:
        item = self._memory.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at is not None and expires_at < time.time():
            del self._memory[key]
            return None
        return value

    def _memory_set(self, key: str, value: Any, expires_at: float | None) -> None:
        self._memory[key] = (value, expires_at)

    def _memory_delete(self, key: str) -> bool:
        return self._memory.pop(key, None) is not None

    def _memory_delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._memory if k.startswith(prefix)]
        for key in keys:
            del self._memory[key]
        return len(keys)
