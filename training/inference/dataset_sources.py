"""Dataset sources for the inference package (Phase R5).

The inference package must be completely independent of the Kaggle / Sentinel
catalog, so the three dataset-facing artefacts — ``metadata.db``,
``historical_context.parquet`` and ``location_index.parquet`` — are snapshotted
from the Dataset Manager at build time and shipped with the package.

:func:`persist_dataset_sources` reads *only* through the manager's public API
(the sole data access path): the metadata database, the spatial index records
and the per-season historical-context availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import DatasetSourceError


@dataclass(frozen=True)
class DatasetSources:
    """Paths of the three dataset artefacts staged for a package."""

    metadata_db: Path
    historical_context: Path
    location_index: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_db": str(self.metadata_db),
            "historical_context": str(self.historical_context),
            "location_index": str(self.location_index),
        }


def persist_dataset_sources(
    manager: Any,
    output_dir: str | Path,
    *,
    resolution: str = "R10m",
    season_months: list[int] | None = None,
    index_type: str | None = None,
) -> DatasetSources:
    """Snapshot the dataset artefacts from ``manager`` into ``output_dir``.

    Args:
        manager: A ``DatasetManager`` instance.
        output_dir: Staging directory for the artefacts.
        resolution: Imagery resolution band for the historical context.
        season_months: Month window for the historical-context availability
            (``None`` = the full catalog).
        index_type: NDVI / EVI index filter (``None`` = both).

    Returns:
        The three written artefact paths.

    Raises:
        DatasetSourceError: When a required artefact cannot be produced.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    db_path = _resolve_db_path(manager)
    if db_path is None or not Path(db_path).exists():
        raise DatasetSourceError(
            "Dataset Manager reports no metadata.db; generate metadata first",
            detail=str(db_path),
        )
    import shutil

    staged_db = out / "metadata.db"
    shutil.copy2(db_path, staged_db)

    location_index = out / "location_index.parquet"
    _persist_location_index(manager, location_index)

    historical_context = out / "historical_context.parquet"
    _persist_historical_context(
        manager, historical_context,
        resolution=resolution, season_months=season_months,
        index_type=index_type,
    )

    return DatasetSources(
        metadata_db=staged_db,
        historical_context=historical_context,
        location_index=location_index,
    )


def _resolve_db_path(manager: Any) -> str | None:
    """Best-effort resolution of the manager's ``metadata.db``.

    R5.2 Task 9: ``DatasetManager`` exposes ``metadata_db_path()`` publicly;
    older / third-party managers may only expose the path via
    ``settings.metadata_db_path()`` or ``metadata_repository.db_path``, so
    those are tried as fallbacks before giving up.
    """
    candidates: list[str] = []
    for attr in ("metadata_db_path", "metadata_db"):
        method = getattr(manager, attr, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if value is not None:
                candidates.append(str(value))
        elif method is not None:
            candidates.append(str(method))
    settings = getattr(manager, "settings", None)
    if settings is not None:
        resolver = getattr(settings, "metadata_db_path", None)
        if callable(resolver):
            try:
                value = resolver()
            except Exception:
                value = None
            if value is not None:
                candidates.append(str(value))
    repo = getattr(manager, "metadata_repository", None)
    if repo is not None:
        path = getattr(repo, "db_path", None)
        if path is not None:
            candidates.append(str(path))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return candidates[0] if candidates else None


def _persist_location_index(manager: Any, path: Path) -> None:
    spatial_index = getattr(manager, "spatial_index", None)
    records = getattr(spatial_index, "records", None)
    if records is None:
        raise DatasetSourceError(
            "Dataset Manager has no spatial index records to persist"
        )
    rows = []
    for record in records():
        rows.append(record.to_dict() if hasattr(record, "to_dict") else dict(record))
    if not rows:
        raise DatasetSourceError(
            "spatial index is empty — nothing to persist as location_index.parquet"
        )
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)


def _persist_historical_context(
    manager: Any,
    path: Path,
    *,
    resolution: str,
    season_months: list[int] | None,
    index_type: str | None,
) -> None:
    availability = manager.get_historical_context(
        window_months=season_months,
        index_type=index_type,
        resolution=resolution,
    )
    per_year = getattr(availability, "per_year", {}) or {}
    years = getattr(availability, "years", []) or []
    rows = [
        {"year": int(year), "record_count": int(per_year.get(str(year), 0))}
        for year in years
    ]
    if not rows and per_year:
        rows = [
            {"year": int(year), "record_count": int(count)}
            for year, count in per_year.items()
        ]
    if not rows:
        raise DatasetSourceError(
            "historical context availability is empty — nothing to persist"
        )
    frame = pd.DataFrame.from_records(rows)
    frame.attrs["dataset_version"] = str(
        getattr(availability, "dataset_version", "") or ""
    )
    frame.attrs["resolution"] = resolution
    frame.to_parquet(path, index=False)


__all__ = ["DatasetSources", "persist_dataset_sources"]
