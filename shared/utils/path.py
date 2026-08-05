"""Path helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing; return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def relposix(path: str | Path, root: str | Path) -> str:
    """POSIX-style relative path of ``path`` within ``root``."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve ``path`` relative to ``base`` (or CWD) into an absolute Path."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base is not None:
        return (Path(base) / candidate).resolve()
    return candidate.resolve()


def iter_unique_by_name_size(paths: Iterable[Path]) -> Iterable[Path]:
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


def is_relative_to(path: str | Path, root: str | Path) -> bool:
    """True when ``path`` is located under ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (accepts 1/yes/true/on)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "yes", "true", "on", "y"}
