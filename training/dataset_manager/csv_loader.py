"""CSV loader: discovery, schema inference, preview and streaming reads.

The loader is the **only** module that reads tabular CSV files — every other
component (validation, metadata generation, future AI/STAM) goes through it.
No filenames are hardcoded; the loader discovers every CSV under a root.

Design notes:

* **Streaming** — heavy operations (row counts, missing-value counts,
  statistics) read the file in bounded chunks instead of loading it fully.
* **Encoding detection** — a lightweight BOM/UTF-8/latin-1 probe runs before
  pandas reads.
* **Single-pass profiling** — :meth:`CSVLoader.profile` computes schema,
  dtypes, missing-value counts, exact row count and numeric statistics in one
  streaming pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import ScanConfig
from .exceptions import CorruptedDatasetError, UnsupportedFormatError
from .interfaces import CSVLoader
from .logger import get_logger
from .models import CSVProfile
from .utils import count_lines_fast, is_csv_path

logger = get_logger("csv_loader")

#: Row cap used for dtype sampling in :meth:`profile` (bounded memory).
_DTYPE_SAMPLE_ROWS = 1_000
#: Chunk size (rows) for streaming passes.
_DEFAULT_CHUNK_ROWS = 100_000


class PandasCSVLoader(CSVLoader):
    """Concrete :class:`CSVLoader` implemented on top of pandas."""

    def __init__(self, config: ScanConfig | None = None) -> None:
        self.config = config or ScanConfig()

    # -- Discovery ------------------------------------------------------------- #

    def discover(self, root: Path) -> list[Path]:
        """Recursively find every CSV file under ``root`` (sorted)."""
        root = root.expanduser().resolve()
        if not root.is_dir():
            return []
        return sorted(
            p for p in root.rglob("*") if p.is_file() and is_csv_path(p)
        )

    # -- Reading --------------------------------------------------------------- #

    def load(
        self,
        path: Path,
        *,
        chunksize: int | None = None,
        columns: list[str] | None = None,
        dtype: dict[str, Any] | None = None,
        encoding: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | Iterator[pd.DataFrame]:
        """Load a CSV file.

        Args:
            path: CSV file to load.
            chunksize: When given, returns an iterator of DataFrames of at
                most this many rows (memory bounded). Otherwise a single
                DataFrame is returned.
            columns: Optional subset of columns to read.
            dtype: Optional dtype overrides.
            encoding: Optional encoding override (auto-detected when None).
            **kwargs: Extra keyword arguments forwarded to ``pd.read_csv``.

        Returns:
            A :class:`pandas.DataFrame`, or an iterator of DataFrames when
            ``chunksize`` is provided.
        """
        self._check_supported(path)
        enc = encoding or self.guess_encoding(path)
        options: dict[str, Any] = {"encoding": enc, "low_memory": False}
        if columns is not None:
            options["usecols"] = columns
        if dtype is not None:
            options["dtype"] = dtype
        options.update(kwargs)

        if chunksize:
            return pd.read_csv(path, chunksize=chunksize, **options)
        return pd.read_csv(path, **options)

    def preview(self, path: Path, n_rows: int = 5) -> pd.DataFrame:
        """Return the first ``n_rows`` rows as a DataFrame."""
        self._check_supported(path)
        return pd.read_csv(path, nrows=max(0, n_rows), encoding=self.guess_encoding(path))

    # -- Profiling ------------------------------------------------------------- #

    def profile(
        self, path: Path, *, chunksize: int = _DEFAULT_CHUNK_ROWS
    ) -> CSVProfile:
        """Compute a full streaming profile of a CSV file.

        Returns a :class:`CSVProfile` with columns, dtypes, exact data row
        count, missing-value counts and numeric statistics.

        Raises:
            CorruptedDatasetError: When the file is empty or unreadable.
        """
        self._check_supported(path)
        encoding = self.guess_encoding(path)
        try:
            reader = pd.read_csv(
                path, chunksize=chunksize, encoding=encoding, low_memory=False
            )
        except Exception as exc:  # noqa: BLE001
            raise CorruptedDatasetError(
                f"Cannot read CSV file: {path.name}", detail=str(exc)
            ) from exc

        first = True
        total_rows = 0
        columns: list[str] = []
        dtypes: dict[str, str] = {}
        missing: dict[str, int] = {}
        num_stats: dict[str, dict[str, float]] = {}
        numeric_cols: list[str] = []

        try:
            for chunk in reader:
                if first:
                    columns = list(chunk.columns)
                    dtypes = {col: str(chunk[col].dtype) for col in chunk.columns}
                    numeric_cols = [
                        col for col in chunk.columns
                        if pd.api.types.is_numeric_dtype(chunk[col])
                    ]
                    missing = {col: 0 for col in chunk.columns}
                    num_stats = {
                        col: {"count": 0.0, "sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf")}
                        for col in numeric_cols
                    }
                    first = False

                rows = len(chunk)
                total_rows += rows
                if rows:
                    isnull = chunk.isna().sum()
                    for col in chunk.columns:
                        missing[col] += int(isnull.get(col, 0))
                    for col in numeric_cols:
                        clean = pd.to_numeric(chunk[col], errors="coerce").dropna()
                        if clean.empty:
                            continue
                        stats = num_stats[col]
                        s = clean.sum()
                        stats["count"] += float(len(clean))
                        stats["sum"] += float(s)
                        stats["sumsq"] += float((clean**2).sum())
                        stats["min"] = min(stats["min"], float(clean.min()))
                        stats["max"] = max(stats["max"], float(clean.max()))
        except Exception as exc:  # noqa: BLE001
            raise CorruptedDatasetError(
                f"Failed while profiling CSV: {path.name}", detail=str(exc)
            ) from exc

        if first:
            # Header-only (or empty-of-data) CSV: still a valid schema, just
            # zero data rows. Re-read the header to populate columns/dtypes.
            try:
                header = pd.read_csv(path, nrows=0, encoding=encoding)
            except Exception as exc:  # noqa: BLE001
                raise CorruptedDatasetError(
                    f"CSV file has no readable content: {path.name}", detail=str(exc)
                ) from exc
            columns = list(header.columns)
            dtypes = {col: str(header[col].dtype) for col in columns}
            missing = {col: 0 for col in columns}
            final_stats = {}

        # Finalise numeric statistics.
        final_stats: dict[str, dict[str, float]] = {}
        for col, stats in num_stats.items():
            n = stats["count"]
            if n == 0:
                continue
            mean = stats["sum"] / n
            variance = max(0.0, (stats["sumsq"] / n) - mean * mean)
            final_stats[col] = {
                "count": n,
                "mean": mean,
                "std": variance**0.5,
                "min": stats["min"],
                "max": stats["max"],
            }

        profile = CSVProfile(
            path=path,
            filename=path.name,
            encoding=encoding,
            row_count=total_rows,
            column_count=len(columns),
            columns=columns,
            dtypes=dtypes,
            missing_values=missing,
            total_missing=sum(missing.values()),
            size_bytes=path.stat().st_size,
            has_header=len(columns) > 0,
        )
        profile.extra = {"statistics": final_stats} if final_stats else {}
        return profile

    def infer_schema(self, path: Path) -> CSVProfile:
        """Alias of :meth:`profile` — infer schema, dtypes and quality."""
        return self.profile(path)

    def detect_missing_values(self, path: Path) -> dict[str, int]:
        """Return ``{column: missing_count}`` for a CSV (streaming)."""
        return self.profile(path).missing_values

    def statistics(self, path: Path) -> dict[str, Any]:
        """Return numeric column statistics (mean/std/min/max/count)."""
        return dict(self.profile(path).extra.get("statistics", {}))

    def row_count(self, path: Path) -> int | None:
        """Fast approximate row count (newline based) for very large files."""
        self._check_supported(path)
        if path.stat().st_size == 0:
            return 0
        return max(0, count_lines_fast(path) - 1)

    # -- Encoding -------------------------------------------------------------- #

    def guess_encoding(self, path: Path) -> str:
        """Detect the file encoding by probing the first bytes.

        BOMs are honoured first; then strict UTF-8; then latin-1 (which never
        fails to decode). UTF-16/32 are only chosen when their BOM is present
        — bare ``decode("utf-16")`` would accept arbitrary latin-1 bytes.
        """
        try:
            with open(path, "rb") as fh:
                head = fh.read(64 * 1024)
        except OSError:
            return "utf-8"
        if head.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if head.startswith((b"\xff\xfe", b"\xfe\xff")):
            return "utf-16"
        if head.startswith((b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")):
            return "utf-32"
        try:
            head.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"

    # -- Internals ------------------------------------------------------------- #

    def _check_supported(self, path: Path) -> None:
        if not Path(path).exists():
            raise UnsupportedFormatError(
                f"CSV file not found: {path}", detail=str(path)
            )
        if not is_csv_path(path):
            raise UnsupportedFormatError(
                f"Not a CSV file: {path}", detail=str(path)
            )
