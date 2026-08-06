"""Metadata loader (Phase R6).

:class:`MetadataLoader` opens the predict-only metadata artefacts shipped in a
release package:

* ``metadata/metadata.db`` — an SQLite snapshot, opened **read-only**;
* ``metadata/historical_context.parquet`` — season availability context;
* ``metadata/location_index.parquet`` — spatial index of locations;
* ``metadata/feature_lookup.parquet`` — feature -> index / type lookup.

Dataframes are served through the runtime cache so repeated access does not
re-read disk.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

from .cache import RuntimeCache
from .config import MetadataConfig, RuntimeConfig
from .exceptions import MetadataLoadError
from .layout import ReleaseLayout

METADATA_ARTIFACTS = (
    "metadata/metadata.db",
    "metadata/historical_context.parquet",
    "metadata/location_index.parquet",
    "metadata/feature_lookup.parquet",
)


@dataclass
class MetadataHealth:
    """Snapshot of the loaded metadata artefacts."""

    loaded: bool
    db_loaded: bool = False
    historical_loaded: bool = False
    location_loaded: bool = False
    feature_lookup_loaded: bool = False
    db_tables: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "db_loaded": self.db_loaded,
            "historical_loaded": self.historical_loaded,
            "location_loaded": self.location_loaded,
            "feature_lookup_loaded": self.feature_lookup_loaded,
            "db_tables": list(self.db_tables),
            "row_counts": dict(self.row_counts),
            "cache": dict(self.cache),
        }


class MetadataLoader:
    """Load the metadata artefacts of a release package.

    Args:
        layout: The release package being loaded.
        config: Validated :class:`RuntimeConfig` (``None`` = defaults).
        cache: Optional shared :class:`RuntimeCache` for dataframes.
    """

    def __init__(
        self,
        layout: ReleaseLayout,
        config: RuntimeConfig | None = None,
        cache: RuntimeCache | None = None,
    ) -> None:
        self.layout = layout
        self.config = config or RuntimeConfig()
        self.metadata_cfg: MetadataConfig = self.config.metadata
        self.cache = cache or RuntimeCache(
            max_bytes=self.metadata_cfg.cache_size_mb * 1024 * 1024,
            max_entries=self.metadata_cfg.cache_max_entries,
        )
        self._conn: sqlite3.Connection | None = None
        self._db_loaded = False
        self._dataframes: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self) -> "MetadataLoader":
        """Open the metadata artefacts.

        Raises:
            MetadataLoadError: When a required artefact is missing or cannot
                be opened.
        """
        if self.metadata_cfg.required:
            for rel in METADATA_ARTIFACTS:
                if not self.layout.exists(rel):
                    raise MetadataLoadError(
                        f"{rel} is missing", detail=str(self.layout.root)
                    )
        self._open_db()
        self._load_dataframes()
        return self

    def _open_db(self) -> None:
        path = self.layout.artifact("metadata/metadata.db")
        if not path.exists():
            self._db_loaded = False
            return
        try:
            self._conn = sqlite3.connect(
                f"file:{path.as_posix()}?mode=ro", uri=True
            )
            self._conn.execute("PRAGMA query_only=ON")
            self._db_loaded = True
        except sqlite3.Error as exc:
            raise MetadataLoadError(
                "failed to open metadata.db", detail=str(path)
            ) from exc

    def _load_dataframes(self) -> None:
        for rel, key in (
            ("metadata/historical_context.parquet", "historical_context"),
            ("metadata/location_index.parquet", "location_index"),
            ("metadata/feature_lookup.parquet", "feature_lookup"),
        ):
            if not self.layout.exists(rel):
                continue
            path = self.layout.artifact(rel)
            try:
                df = self.cache.get(key)
                if df is None:
                    import pandas as pd

                    df = pd.read_parquet(path)
                    self.cache.set(key, df)
                self._dataframes[key] = df
            except Exception as exc:  # noqa: BLE001 - surface the read failure
                if self.metadata_cfg.feature_lookup_required and key == "feature_lookup":
                    raise MetadataLoadError(
                        "failed to read feature_lookup.parquet", detail=str(path)
                    ) from exc
                self._dataframes[key] = None

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The read-only SQLite connection (or ``None``)."""
        return self._conn

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        """Run a read-only query against ``metadata.db``.

        Raises:
            MetadataLoadError: When the database is not loaded or the query
                fails.
        """
        if self._conn is None:
            raise MetadataLoadError(
                "metadata.db is not loaded", detail=str(self.layout.root)
            )
        try:
            cursor = self._conn.execute(sql, tuple(params))
            return list(cursor.fetchall())
        except sqlite3.Error as exc:
            raise MetadataLoadError("metadata.db query failed", detail=sql) from exc

    def tables(self) -> list[str]:
        if self._conn is None:
            return []
        try:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return [row[0] for row in rows]
        except sqlite3.Error:
            return []

    def historical(self) -> Any:
        """The ``historical_context`` dataframe (or ``None``)."""
        return self._dataframes.get("historical_context")

    def locations(self) -> Any:
        """The ``location_index`` dataframe (or ``None``)."""
        return self._dataframes.get("location_index")

    def feature_lookup(self) -> Any:
        """The ``feature_lookup`` dataframe (or ``None``)."""
        return self._dataframes.get("feature_lookup")

    def load_config(self) -> dict[str, Any]:
        return self._config()

    def load_metadata(self) -> dict[str, Any]:
        info = {
            "db_loaded": self._db_loaded,
            "tables": self.tables(),
            "dataframes": {
                key: None if df is None else len(df)
                for key, df in self._dataframes.items()
            },
        }
        return info

    def _config(self) -> dict[str, Any]:
        return {
            "required": self.metadata_cfg.required,
            "feature_lookup_required": self.metadata_cfg.feature_lookup_required,
            "artifacts": [rel for rel in METADATA_ARTIFACTS if self.layout.exists(rel)],
        }

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health(self) -> MetadataHealth:
        return MetadataHealth(
            loaded=self._db_loaded or bool(self._dataframes),
            db_loaded=self._db_loaded,
            historical_loaded=self.historical() is not None,
            location_loaded=self.locations() is not None,
            feature_lookup_loaded=self.feature_lookup() is not None,
            db_tables=self.tables(),
            row_counts={
                key: (None if df is None else len(df))
                for key, df in self._dataframes.items()
            },
            cache=self.cache.info(),
        )

    def unload(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - defensive
                pass
        self._conn = None
        self._db_loaded = False
        self._dataframes.clear()
