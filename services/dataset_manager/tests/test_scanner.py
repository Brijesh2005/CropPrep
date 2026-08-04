"""Tests for the dataset scanner (classification, caching, parallel scan)."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.dataset_manager.exceptions import DatasetNotFoundError
from services.dataset_manager.models import IndexType, Resolution
from services.dataset_manager.scanner import DatasetScanner


def test_scan_classifies_tree(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    scanner = DatasetScanner()
    inventory = scanner.scan(root, use_cache=False)

    counts = inventory.counts()
    assert counts["csv"] == 1
    assert counts["geotiff"] == 3
    assert counts["ndvi"] == 2
    assert counts["evi"] == 1
    assert counts["r10m"] == 2  # the two tiffs under R10m/


def test_scan_classification_details(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    inventory = DatasetScanner().scan(root, use_cache=False)
    # The R10m/NDVI file carries both index and resolution tags.
    ndvi = next(
        e for e in inventory.entries
        if e.index_type is IndexType.NDVI and e.resolution is Resolution.R10M
    )
    assert ndvi.year == 2019
    assert ndvi.relative_path.startswith("2019_images")
    # The CSV is tagged with a year too.
    csv_entry = next(e for e in inventory.entries if e.category.value == "csv")
    assert csv_entry.year == 2019


def test_scan_missing_root_raises(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError):
        DatasetScanner().scan(tmp_path / "does-not-exist")


def test_scan_cache_hit(synthetic_dataset: Path):
    from services.dataset_manager.cache_manager import CacheManager

    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    cache = CacheManager(enabled=True, db_path=synthetic_dataset / ".dm" / "cache.db")
    scanner = DatasetScanner(cache=cache)
    first = scanner.scan(root, use_cache=True)
    assert first.source == "scan"
    second = scanner.scan(root, use_cache=True)
    assert second.source == "cache"
    assert len(second.entries) == len(first.entries)


def test_scan_cache_invalidates_on_change(synthetic_dataset: Path):
    from services.dataset_manager.cache_manager import CacheManager

    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    cache = CacheManager(enabled=True, db_path=synthetic_dataset / ".dm" / "cache2.db")
    scanner = DatasetScanner(cache=cache)
    scanner.scan(root, use_cache=True)
    assert scanner.scan(root, use_cache=True).source == "cache"

    (root / "new_file.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    fresh = scanner.scan(root, use_cache=True)
    assert fresh.source == "scan"
    assert fresh.counts()["csv"] == 2


def test_scan_parallel_matches_serial(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    from services.dataset_manager.config import ScanConfig

    serial = DatasetScanner(ScanConfig(workers=1)).scan(root, use_cache=False)
    parallel = DatasetScanner(ScanConfig(workers=8)).scan(root, use_cache=False)
    assert {e.relative_path for e in serial.entries} == {
        e.relative_path for e in parallel.entries
    }


def test_scan_hash_files(synthetic_dataset: Path):
    from services.dataset_manager.config import ScanConfig

    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    inventory = DatasetScanner(ScanConfig(hash_files=True)).scan(root, use_cache=False)
    assert all(e.sha256 for e in inventory.entries if e.category.value == "csv")
