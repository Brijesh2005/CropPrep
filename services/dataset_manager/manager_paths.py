"""Filesystem layout helpers for the Dataset Manager.

Ensures the managed directory tree described in the SDD exists:

::

    <dataset_root>/
    ├── raw/                      # canonical copy of downloads
    │   └── <catalog_name>/       # e.g. kaggle-crop-yield
    ├── processed/                # derived / cleaned datasets
    └── .cropfusion/              # internal state (sqlite stores, cache)
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .logger import get_logger

logger = get_logger("paths")


def ensure_state_dirs(settings: Settings) -> None:
    """Create the dataset root layout if it does not exist yet."""
    for directory in (
        settings.dataset_root,
        settings.raw_root,
        settings.processed_root,
        settings.state_root,
        settings.catalog_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "Ensured dataset layout",
        extra={
            "dataset_root": str(settings.dataset_root),
            "raw_root": str(settings.raw_root),
            "state_root": str(settings.state_root),
        },
    )


def relative_to_root(path: str | Path, root: str | Path) -> str:
    """Return a POSIX relative path for ``path`` under ``root``.

    Falls back to the basename when ``path`` is outside ``root``.
    """
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name
