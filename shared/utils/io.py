"""File I/O, walking, copying and format-detection helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..constants import CHUNK_SIZE, CSV_SUFFIXES, EXCLUDE_DIRS, RASTER_SUFFIXES, TIFF_MAGIC


def count_lines_fast(path: str | Path, chunk_size: int = CHUNK_SIZE) -> int:
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
    return head[:4] in TIFF_MAGIC


def is_csv_path(path: str | Path) -> bool:
    """True when the file has a CSV-like extension."""
    return Path(path).suffix.lower() in CSV_SUFFIXES


def is_geotiff_path(path: str | Path) -> bool:
    """True when the file has a raster-like extension and TIFF magic."""
    suffix = Path(path).suffix.lower()
    return suffix in RASTER_SUFFIXES and is_geotiff_bytes(path)


def safe_float(value: object) -> float | None:
    """Parse a value as float, returning None on failure."""
    try:
        if value is None or value == "":
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> int | None:
    """Parse a value as int, returning None on failure."""
    try:
        if value is None or value == "":
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def walk_files(
    root: str | Path, exclude_dirs: Iterable[str] = EXCLUDE_DIRS
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
    root: str | Path, exclude_dirs: Iterable[str] = EXCLUDE_DIRS
) -> tuple[int, int, float]:
    """Cheap signature of a directory tree: ``(file_count, total_size, max_mtime)``.

    Used to invalidate caches without hashing every file. Internal state
    directories (``.cropfusion`` etc.) are excluded so that writing SQLite
    state does not invalidate the inventory cache.
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
    chunk_size: int = CHUNK_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Stream-copy ``src`` to ``dst``, invoking ``progress(so_far, total)``.

    Returns the number of bytes copied.
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
