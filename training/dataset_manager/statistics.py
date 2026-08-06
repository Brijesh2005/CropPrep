"""Aggregate dataset statistics across tabular and image datasets.

:func:`compute_statistics` combines the tabular provider's numeric column
statistics with the image catalog's year / index / resolution counts into one
:class:`DatasetStatistics` object used by reports, the CLI and the manager's
``statistics()`` API.

Everything is computed through the provider layer — no direct file reads.
"""

from __future__ import annotations

from typing import Any

from .logger import get_logger
from .models import DatasetStatistics, FileCategory

logger = get_logger("statistics")


def compute_statistics(
    *,
    tabular_provider: Any | None = None,
    image_provider: Any | None = None,
) -> DatasetStatistics:
    """Compute :class:`DatasetStatistics` from the provider layer.

    Args:
        tabular_provider: Optional :class:`TabularProvider` (None skips the
            tabular portion).
        image_provider: Optional :class:`ImageProvider` (None skips the image
            portion).

    Returns:
        A populated :class:`DatasetStatistics`.
    """
    stats = DatasetStatistics()

    if tabular_provider is not None:
        try:
            names = tabular_provider.names()
        except Exception:  # noqa: BLE001 - best effort
            names = []
        total_rows = 0
        for name in names:
            try:
                column_stats = tabular_provider.statistics(name)
            except Exception:  # noqa: BLE001 - best effort
                column_stats = {}
            stats.tabular[name] = column_stats
            try:
                schema = tabular_provider.schema(name)
                row_count = schema.get("row_count") or 0
            except Exception:  # noqa: BLE001 - best effort
                row_count = 0
            stats.tabular_row_counts[name] = row_count
            total_rows += row_count
        stats.total_tabular_rows = total_rows

    if image_provider is not None:
        try:
            catalog = image_provider.catalog()
        except Exception as exc:  # noqa: BLE001 - not materialised yet
            logger.warning("Image catalog unavailable for statistics", extra={"error": str(exc)})
            catalog = None
        if catalog is not None:
            rasters = [e for e in catalog.entries if e.category is FileCategory.GEOTIFF]
            for year, count in _count_by_year(rasters).items():
                stats.images_by_year[year] = count
            stats.images_by_index = {
                "NDVI": len(catalog.ndvi),
                "EVI": len(catalog.evi),
            }
            stats.images_by_resolution = {
                r.value: sum(
                    1 for e in rasters if e.resolution is _resolution_for(r.value)
                )
                for r in _distinct_resolutions(rasters)
            }
            stats.total_images = len(rasters)

    return stats


def _count_by_year(entries: list[Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for entry in entries:
        if entry.year is None:
            continue
        out[entry.year] = out.get(entry.year, 0) + 1
    return out


def _distinct_resolutions(entries: list[Any]) -> list[Any]:
    seen: list[Any] = []
    for entry in entries:
        if entry.resolution not in seen:
            seen.append(entry.resolution)
    return seen


def _resolution_for(value: str) -> Any:
    from .models import Resolution

    return {"R10m": Resolution.R10M, "R20m": Resolution.R20M}.get(value, Resolution.UNKNOWN)
