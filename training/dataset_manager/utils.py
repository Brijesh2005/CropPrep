"""Assorted helpers shared across the Dataset Manager.

Everything here is intentionally free of package-internal imports (only the
standard library, plus lazy optional imports) so it can be used from any
module without introducing circular dependencies.

Highlights:

* :func:`sha256_file` — streaming content hash (memory bounded).
* :func:`count_lines_fast` — streaming newline count for CSV row estimates.
* :func:`run_parallel` — thread-pool map for parallel directory scanning.
* :func:`is_geotiff_bytes` — cheap magic-byte check that does not require GDAL.
* :func:`classify_index_type` / :func:`classify_resolution` /
  :func:`extract_year_from_path` / :func:`parse_observation_date` — naming
  conventions used by the scanner and the metadata generator.
"""

from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from .models import IndexType, Resolution

# --------------------------------------------------------------------------- #
# Hashing & file utilities
# --------------------------------------------------------------------------- #

T = TypeVar("T")
R = TypeVar("R")

_CHUNK = 1 << 20  # 1 MiB

#: TIFF magic bytes (little endian "II*\0" and big endian "MM\0*").
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")


def sha256_file(path: str | Path, chunk_size: int = _CHUNK) -> str:
    """Return the SHA-256 of a file, streaming it in bounded chunks.

    Args:
        path: File to hash.
        chunk_size: Read chunk size in bytes.

    Returns:
        Lower-case hex digest of the file contents.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_lines_fast(path: str | Path, chunk_size: int = _CHUNK) -> int:
    """Count newline characters in a file without loading it fully.

    Used for fast CSV row estimates. Returns the number of ``\\n`` bytes,
    which is equal to the number of lines.
    """
    count = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            count += block.count(b"\n")
    return count


def human_size(num_bytes: float) -> str:
    """Format a byte count as a human readable string (e.g. ``"4.2 MiB"``)."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"  # pragma: no cover - unreachable


def is_geotiff_bytes(path: str | Path) -> bool:
    """Cheap TIFF detection via magic bytes (no GDAL required)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        return False
    return head[:4] in _TIFF_MAGIC


def is_csv_path(path: str | Path) -> bool:
    """True when the file has a CSV-like extension."""
    return Path(path).suffix.lower() in {".csv", ".txt"}


def is_geotiff_path(path: str | Path) -> bool:
    """True when the file has a raster-like extension and TIFF magic."""
    suffix = Path(path).suffix.lower()
    return suffix in {".tif", ".tiff"} and is_geotiff_bytes(path)


def safe_float(value: Any) -> float | None:
    """Parse a value as float, returning None on failure."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    """Parse a value as int, returning None on failure."""
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Parallel helpers
# --------------------------------------------------------------------------- #


def run_parallel(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    workers: int | None = None,
    raise_on_error: bool = False,
) -> list[R]:
    """Apply ``worker`` to every item in ``items`` using a thread pool.

    The order of results matches the order of ``items``. When
    ``raise_on_error`` is False (default) a worker exception is captured as
    the result for that slot; when True the exception is re-raised.

    Args:
        items: Iterable of inputs (materialised once).
        worker: Callable applied to each item.
        workers: Thread count. Defaults to ``min(32, cpu_count + 4)``.
        raise_on_error: Re-raise worker exceptions instead of capturing them.
    """
    pool_size = workers or min(32, (os.cpu_count() or 1) + 4)
    if not items:
        return []
    results: list[R] = [None] * len(items)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, pool_size)) as pool:
        future_map = {pool.submit(worker, item): i for i, item in enumerate(items)}
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - captured per design
                if raise_on_error:
                    raise
                results[index] = exc  # type: ignore[assignment]
    return results


#: Directory names always skipped during discovery (internal state / VCS).
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".cropfusion", ".cache", ".git", "__pycache__", ".idea", ".venv", "venv"}
)


def walk_files(
    root: str | Path, exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS
) -> Iterator[Path]:
    """Yield every file under ``root`` (absolute paths), walking recursively.

    Args:
        root: Directory to walk.
        exclude_dirs: Directory *names* to prune during the walk (matched
            case-insensitively so ``.CropFusion`` is excluded too).
    """
    root_path = Path(root)
    if not root_path.exists():
        return
    excluded = {name.lower() for name in (exclude_dirs or ())}
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in excluded]
        for name in filenames:
            yield Path(dirpath) / name


def tree_signature(
    root: str | Path, exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS
) -> tuple[int, int, float]:
    """Cheap signature of a directory tree: ``(file_count, total_size, max_mtime)``.

    Used to invalidate the scan cache without hashing every file. Internal
    state directories (``.cropfusion`` etc.) are excluded so that writing
    SQLite state does not invalidate the inventory cache.

    Args:
        root: Directory to inspect.
        exclude_dirs: Directory names to prune (see :func:`walk_files`).
    """
    count = 0
    total_size = 0
    max_mtime = 0.0
    for path in walk_files(root, exclude_dirs=exclude_dirs):
        try:
            st = path.stat()
        except OSError:
            continue
        count += 1
        total_size += st.st_size
        if st.st_mtime > max_mtime:
            max_mtime = st.st_mtime
    return (count, total_size, max_mtime)


def copy_file_with_progress(
    src: str | Path,
    dst: str | Path,
    *,
    chunk_size: int = _CHUNK,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Stream-copy ``src`` to ``dst``, invoking ``progress(so_far, total)``.

    Returns the number of bytes copied. Uses ``shutil.copyfileobj`` semantics
    but with explicit byte accounting for progress reporting.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    total = src_path.stat().st_size
    copied = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "rb") as reader, open(dst_path, "wb") as writer:
        while True:
            block = reader.read(chunk_size)
            if not block:
                break
            writer.write(block)
            copied += len(block)
            if progress is not None:
                progress(copied, total)
    return copied


# --------------------------------------------------------------------------- #
# Naming-convention classifiers
# --------------------------------------------------------------------------- #

_YEAR_RE = re.compile(r"(?<![\d])(20\d{2}|19\d{2})(?![\d])")
_OBS_DATE_RE = re.compile(
    r"(?<![\d])(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?![\d])"
)
_R10_RE = re.compile(r"R\s?10\s?m", re.IGNORECASE)
_R20_RE = re.compile(r"R\s?20\s?m", re.IGNORECASE)


def classify_index_type(name: str) -> IndexType:
    """Detect the vegetation index from a path segment or file name."""
    upper = name.upper()
    if "NDVI" in upper:
        return IndexType.NDVI
    if "EVI" in upper:
        return IndexType.EVI
    return IndexType.NONE


def classify_index_type_from_path(path: str | Path) -> IndexType:
    """Detect the index by inspecting every path segment of ``path``."""
    parts = Path(path).parts
    for part in reversed(parts):
        detected = classify_index_type(part)
        if detected is not IndexType.NONE:
            return detected
    return IndexType.NONE


def classify_resolution(name: str) -> Resolution:
    """Detect the resolution band (R10m/R20m) from a path segment or name."""
    if _R10_RE.search(name):
        return Resolution.R10M
    if _R20_RE.search(name):
        return Resolution.R20M
    # Sentinel-2 naming convention: S2A_..._10m_... / _20m_
    if re.search(r"(?<![Rr])\b10m\b", name, re.IGNORECASE):
        return Resolution.R10M
    if re.search(r"(?<![Rr])\b20m\b", name, re.IGNORECASE):
        return Resolution.R20M
    return Resolution.UNKNOWN


def classify_resolution_from_path(path: str | Path) -> Resolution:
    """Detect the resolution by inspecting every path segment of ``path``."""
    parts = Path(path).parts
    for part in parts:
        detected = classify_resolution(part)
        if detected is not Resolution.UNKNOWN:
            return detected
    return Resolution.UNKNOWN


def extract_year_from_path(path: str | Path) -> int | None:
    """Extract the first 4-digit year (19xx/20xx) found in the path."""
    match = _YEAR_RE.search(str(path))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1950 <= year <= 2100 else None


def parse_observation_date(path: str | Path) -> date | None:
    """Parse an ``YYYY[_-]?MM[_-]?DD`` date from a file name, if present."""
    match = _OBS_DATE_RE.search(Path(path).name)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def iter_unique_by_name_size(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield paths, skipping duplicates that share ``(name, size_bytes)``."""
    seen: set[tuple[str, int]] = set()
    for path in paths:
        try:
            st = path.stat()
        except OSError:
            continue
        key = (path.name, st.st_size)
        if key in seen:
            continue
        seen.add(key)
        yield path
