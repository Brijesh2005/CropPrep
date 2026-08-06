"""Tests for :func:`compute_statistics` — the R2.2 aggregate statistics.

Verifies tabular row counts / column stats and image year / index / resolution
totals computed through the provider layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training.dataset_manager.statistics import compute_statistics


def test_statistics_tabular_totals(r22_manager_factory):
    manager = r22_manager_factory()
    stats = manager.statistics()
    assert stats.total_tabular_rows == 3
    assert stats.tabular_row_counts["crop_yield"] == 3
    assert "crop_yield" in stats.tabular
    yield_stats = stats.tabular["crop_yield"]["yield_kg"]
    assert yield_stats["count"] == 3
    assert yield_stats["min"] == 3100.0
    assert yield_stats["max"] == 5400.0


def test_statistics_image_counts(r22_manager_factory):
    manager = r22_manager_factory()
    stats = manager.statistics()
    assert stats.total_images == 3  # 2019 NDVI + 2019 EVI + 2020 NDVI
    assert stats.images_by_year == {2019: 2, 2020: 1}
    assert stats.images_by_index == {"NDVI": 2, "EVI": 1}
    assert stats.images_by_resolution == {"R10m": 3}


def test_statistics_to_dict(r22_manager_factory):
    manager = r22_manager_factory()
    data = manager.statistics().to_dict()
    assert data["total_images"] == 3
    assert data["images_by_year"] == {"2019": 2, "2020": 1}
    assert "generated_at" in data


def test_statistics_without_image_provider():
    stats = compute_statistics(tabular_provider=None, image_provider=None)
    assert stats.total_images == 0
    assert stats.total_tabular_rows == 0
    assert stats.images_by_year == {}


def test_statistics_ignores_non_geotiff_catalog_entries(r22_manager_factory):
    manager = r22_manager_factory()
    stats = manager.statistics()
    # The CSV must never count towards image totals.
    assert 3 == stats.total_images
    assert set(stats.images_by_resolution) == {"R10m"}
