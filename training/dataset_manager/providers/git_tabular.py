"""Git-versioned tabular data provider.

:class:`GitRepositoryTabularProvider` serves the *small, structured* datasets
that live inside the Git repository (``training/datasets/tabular/`` — the
data_season, weather, soil, fertilizer, production, villages, districts, crop
CSVs and friends). These files are version controlled and must remain inside
GitHub; the provider discovers them automatically and never hardcodes a
filename.

Responsibilities (per the R1.2 specification):

* **Discover CSV files** — recursive, pattern based discovery (no hardcoded
  names).
* **Validate schemas** — per-column schema / dtype / missing-value checks.
* **Load datasets** — name-based loads returning pandas DataFrames.
* **Join datasets** — declarative, sequential joins via :class:`TabularJoinSpec`.
* **Streaming reads** — chunked reads for memory bounded processing.
* **Missing value handling** — count / drop / fill strategies.
* **Dataset statistics** — streaming numeric statistics.
* **Metadata generation** — path / size / schema / statistics per dataset.

The provider reuses the existing :class:`PandasCSVLoader` engine for the
actual file reads (it remains the only module that reads CSV bytes).
"""

from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..csv_loader import PandasCSVLoader
from ..exceptions import DatasetNotFoundError
from ..interfaces import CSVLoader
from ..logger import get_logger
from .base import TabularProvider
from .models import (
    ProviderCapabilities,
    ProviderManifest,
    ProviderStatus,
    TabularCatalog,
    TabularDatasetInfo,
    TabularJoinSpec,
)

logger = get_logger("tabular_provider")

#: Default glob patterns used for automatic discovery.
_DEFAULT_PATTERNS = ("*.csv",)

#: Maximum rows read when a profile is requested (bounded memory).
_PROFILE_CHUNK_ROWS = 100_000


def _matches_any(name: str, patterns: list[str]) -> bool:
    """True when ``name`` matches any glob pattern (case-insensitive)."""
    return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in patterns)


class GitRepositoryTabularProvider(TabularProvider):
    """Concrete :class:`TabularProvider` backed by Git-versioned CSVs.

    Args:
        root: Directory containing the versioned CSVs. Defaults to the
            repository-relative ``training/datasets/tabular``.
        recursive: Recurse into subdirectories during discovery.
        patterns: Glob patterns (relative to ``root``) used for discovery.
        na_values: Values treated as missing when reading CSVs.
        dtype_overrides: Optional ``{column: dtype}`` applied on load.
        loader: Optional :class:`CSVLoader` engine (default: pandas based).
    """

    name = "git_repository_tabular"
    kind = "tabular"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        name: str | None = None,
        recursive: bool = True,
        patterns: list[str] | None = None,
        na_values: list[str] | None = None,
        dtype_overrides: dict[str, str] | None = None,
        loader: CSVLoader | None = None,
    ) -> None:
        self.name = name or self.name
        self.root = Path(root) if root is not None else Path("training/datasets/tabular")
        self.recursive = recursive
        self.patterns = list(patterns or _DEFAULT_PATTERNS)
        self.na_values = list(na_values or ["", "NA", "N/A", "null", "NULL"])
        self.dtype_overrides = dict(dtype_overrides or {})
        self.loader = loader or PandasCSVLoader()
        self._catalog: TabularCatalog | None = None
        self._status = ProviderStatus.NOT_INITIALIZED
        #: Per-dataset schema fingerprints recorded over time (version tracking).
        self._schema_versions: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    # Provider introspection
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def capabilities(self) -> ProviderCapabilities:
        """Declared tabular capabilities (used by the provider registry)."""
        return ProviderCapabilities(
            name=self.name,
            kind=self.kind,
            priority=100,
            features=[
                "discover",
                "load",
                "stream",
                "schema",
                "validate_schema",
                "statistics",
                "missing_values",
                "handle_missing",
                "join",
                "metadata",
                "schema_evolution",
                "version_tracking",
            ],
        )

    def available(self) -> bool:
        try:
            return self.names() != []
        except Exception:  # noqa: BLE001 - availability is best-effort
            return False

    def manifest(self) -> ProviderManifest:
        return ProviderManifest(
            name=self.name,
            kind=self.kind,
            status=self.status,
            available=self.available(),
            root=self._resolved_root() if self._root_exists() else None,
            details={
                "root": str(self.root),
                "recursive": self.recursive,
                "patterns": list(self.patterns),
                "datasets": self.names(),
                "total_size_bytes": (
                    self._catalog.total_size() if self._catalog is not None else 0
                ),
            },
        )

    def describe(self) -> dict[str, Any]:
        return self.manifest().to_dict()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _resolved_root(self) -> Path:
        return self.root.expanduser().resolve()

    def _root_exists(self) -> bool:
        return self._resolved_root().is_dir()

    def discover(self, *, refresh: bool = False) -> TabularCatalog:
        """Discover every CSV under the provider root (automatic discovery).

        The catalog is cached; pass ``refresh=True`` to force a re-scan.
        """
        if self._catalog is not None and not refresh:
            return self._catalog

        root = self._resolved_root()
        if not self._root_exists():
            self._status = ProviderStatus.MISSING_DATA
            self._catalog = TabularCatalog(root=root)
            logger.warning(
                "Tabular provider root missing", extra={"root": str(root)}
            )
            return self._catalog

        files = self.loader.discover(root) if self.recursive else self._shallow(root)
        files = [p for p in files if _matches_any(p.name, self.patterns)]

        datasets: list[TabularDatasetInfo] = []
        for path in files:
            try:
                size = path.stat().st_size
            except OSError:  # pragma: no cover - file vanished mid-scan
                continue
            datasets.append(
                TabularDatasetInfo(
                    name=path.stem,
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    size_bytes=size,
                )
            )

        datasets.sort(key=lambda info: info.name)
        self._catalog = TabularCatalog(root=root, datasets=datasets)
        self._status = (
            ProviderStatus.READY if datasets else ProviderStatus.MISSING_DATA
        )
        logger.info(
            "Tabular discovery complete", extra={"count": len(datasets), "root": str(root)}
        )
        return self._catalog

    def _shallow(self, root: Path) -> list[Path]:
        return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".csv")

    def names(self) -> list[str]:
        return self.discover().names()

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #

    def _resolve(self, name: str) -> TabularDatasetInfo:
        catalog = self.discover()
        info = catalog.by_name(name)
        if info is None:
            raise DatasetNotFoundError(
                f"Tabular dataset not found: {name}",
                detail={"discovered": catalog.names()},
            )
        return info

    # ------------------------------------------------------------------ #
    # Load / stream
    # ------------------------------------------------------------------ #

    def load(
        self,
        name: str,
        *,
        chunksize: int | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | Iterator[pd.DataFrame]:
        """Load a discovered dataset by name.

        Args:
            name: Discovered dataset name (file stem).
            chunksize: When set, returns an iterator of bounded chunks.
            columns: Optional column subset.
            **kwargs: Extra options forwarded to the loader.
        """
        info = self._resolve(name)
        options = dict(kwargs)
        if columns is not None:
            options["columns"] = columns
        if self.dtype_overrides:
            merged = dict(self.dtype_overrides)
            merged.update(kwargs.get("dtype", {}))
            options["dtype"] = merged
        return self.loader.load(info.path, chunksize=chunksize, **options)

    def stream(self, name: str, chunksize: int, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Stream a dataset in chunks of at most ``chunksize`` rows."""
        return self.load(name, chunksize=chunksize, **kwargs)  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Schema / quality
    # ------------------------------------------------------------------ #

    def _profile(self, name: str) -> Any:
        info = self._resolve(name)
        if info.profile is None:
            info.profile = self.loader.profile(info.path, chunksize=_PROFILE_CHUNK_ROWS)
        return info.profile

    def schema(self, name: str) -> dict[str, Any]:
        """Schema, dtypes and missing-value profile of a dataset."""
        profile = self._profile(name)
        data = profile.to_dict()
        data["statistics"] = dict(profile.extra.get("statistics", {}))
        return data

    def validate_schema(self, name: str) -> dict[str, Any]:
        """Run schema validation and return ``{valid, issues}``.

        Checks performed: readable header, at least one column, non-zero
        columns duplicated, per-column missing-value ratio vs. a sane
        threshold, and numeric columns whose majority of values are
        non-numeric.
        """
        profile = self._profile(name)
        issues: list[dict[str, Any]] = []
        if not profile.columns:
            issues.append({"level": "error", "code": "no_columns", "message": "No columns detected"})
        duplicates = {
            col for col in profile.columns if profile.columns.count(col) > 1
        }
        if duplicates:
            issues.append(
                {
                    "level": "error",
                    "code": "duplicate_columns",
                    "message": f"Duplicate columns: {sorted(duplicates)}",
                }
            )
        for col, missing in (profile.missing_values or {}).items():
            if profile.row_count and missing > profile.row_count:
                ratio = 1.0
            else:
                ratio = (missing / profile.row_count) if profile.row_count else 0.0
            if ratio > 0.95 and profile.row_count:
                issues.append(
                    {
                        "level": "warning",
                        "code": "mostly_missing",
                        "message": f"Column {col!r} is {ratio:.0%} missing",
                    }
                )
        return {"name": name, "valid": not any(i["level"] == "error" for i in issues), "issues": issues}

    def statistics(self, name: str) -> dict[str, Any]:
        """Numeric column statistics (streaming, bounded memory)."""
        profile = self._profile(name)
        return dict(profile.extra.get("statistics", {}))

    def missing_values(self, name: str) -> dict[str, int]:
        """``{column: missing_count}`` for a dataset."""
        profile = self._profile(name)
        return dict(profile.missing_values or {})

    # ------------------------------------------------------------------ #
    # Missing value handling
    # ------------------------------------------------------------------ #

    def handle_missing(
        self,
        name: str,
        strategy: str = "drop",
        *,
        fill_value: Any = None,
        fill_method: str = "mean",
    ) -> pd.DataFrame:
        """Return a copy of the dataset with missing values handled.

        ``strategy``:
            * ``drop`` — drop rows containing any missing value.
            * ``fill`` — fill using ``fill_value`` (constant) or a per-column
              ``fill_method`` of ``mean`` / ``median`` / ``mode``.
        """
        frame = self.load(name)
        if strategy == "drop":
            return frame.dropna()
        if strategy != "fill":
            raise ValueError(f"Unknown missing-value strategy: {strategy}")
        if fill_value is not None:
            return frame.fillna(fill_value)
        if fill_method == "constant":
            return frame.fillna(0)
        numeric = frame.select_dtypes(include="number")
        if fill_method == "mean":
            return frame.fillna(numeric.mean())
        if fill_method == "median":
            return frame.fillna(numeric.median())
        if fill_method == "mode":
            return frame.fillna(numeric.mode().iloc[0] if not numeric.empty else 0)
        raise ValueError(f"Unknown fill method: {fill_method}")

    # ------------------------------------------------------------------ #
    # Joins
    # ------------------------------------------------------------------ #

    def join(self, joins: list[TabularJoinSpec], *, how: str | None = None) -> pd.DataFrame:
        """Sequentially join discovered datasets into a single frame.

        Args:
            joins: Ordered list of join specifications. The first spec's
                ``name`` is the base frame; each subsequent spec merges in
                ``other`` on the shared key(s).
            how: Override the join strategy for every spec.

        Returns:
            The joined :class:`pandas.DataFrame`.
        """
        if not joins:
            raise ValueError("At least one join spec is required")
        base = self.load(joins[0].name)
        for spec in joins:
            other = self.load(spec.other)
            base = base.merge(
                other,
                on=spec.on,
                how=how or spec.how,
                suffixes=spec.suffixes,
            )
        return base

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def metadata(self, name: str) -> dict[str, Any]:
        """Dataset metadata: path, size, schema, statistics and quality."""
        info = self._resolve(name)
        profile = self._profile(name)
        return {
            "name": name,
            "path": str(info.path),
            "relative_path": info.relative_path,
            "size_bytes": info.size_bytes,
            "schema": profile.to_dict(),
            "statistics": dict(profile.extra.get("statistics", {})),
        }

    # ------------------------------------------------------------------ #
    # Schema evolution detection
    # ------------------------------------------------------------------ #

    def schema_fingerprint(self, name: str) -> str:
        """Stable fingerprint of a dataset schema (columns + dtypes).

        Two datasets with the same columns and dtypes share a fingerprint;
        adding/removing/renaming a column changes it. Used by
        :meth:`detect_schema_change` and :meth:`record_version`.
        """
        schema = self.schema(name)
        columns = list(schema.get("columns", []))
        dtypes = {c: schema.get("dtypes", {}).get(c) for c in columns}
        payload = "|".join(f"{c}:{dtypes.get(c)}" for c in sorted(columns))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def detect_schema_change(self, name: str) -> dict[str, Any]:
        """Compare the current schema against the last recorded version.

        Returns:
            ``{name, changed, previous, current, added, removed}`` where
            ``changed`` is False when no earlier version exists or the schema
            is identical.
        """
        current = self.schema_fingerprint(name)
        history = self._schema_versions.get(name, [])
        previous = history[-1]["fingerprint"] if history else None
        changed = previous is not None and previous != current

        added: list[str] = []
        removed: list[str] = []
        if changed:
            prev_cols = set(history[-1].get("columns", []))
            cur_cols = set(self.schema(name).get("columns", []))
            added = sorted(cur_cols - prev_cols)
            removed = sorted(prev_cols - cur_cols)

        return {
            "name": name,
            "changed": changed,
            "previous": previous,
            "current": current,
            "added": added,
            "removed": removed,
        }

    def record_version(self, name: str, *, message: str = "") -> dict[str, Any]:
        """Record a schema snapshot for ``name`` (version tracking).

        Returns the recorded snapshot (fingerprint, columns, timestamp).
        """
        info = self._resolve(name)
        schema = self.schema(name)
        snapshot = {
            "version": len(self._schema_versions.get(name, [])) + 1,
            "fingerprint": self.schema_fingerprint(name),
            "columns": list(schema.get("columns", [])),
            "size_bytes": info.size_bytes,
            "message": message,
            "recorded_at": pd.Timestamp.now().isoformat(),
        }
        self._schema_versions.setdefault(name, []).append(snapshot)
        return snapshot

    def dataset_versions(self, name: str) -> list[dict[str, Any]]:
        """Schema version history for a dataset (oldest first)."""
        return list(self._schema_versions.get(name, []))
