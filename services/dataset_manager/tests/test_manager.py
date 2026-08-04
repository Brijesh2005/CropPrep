"""Integration tests for the DatasetManager facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.dataset_manager.exceptions import DatasetNotFoundError
from services.dataset_manager.models import IndexType


def test_download_materializes_to_catalog(manager_factory, tmp_path: Path):
    from services.dataset_manager.downloader import KaggleDownloader

    source = tmp_path / "download_src"
    (source / "2019_images").mkdir(parents=True)
    (source / "2019_images" / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    class FakeDownloader(KaggleDownloader):
        def download(self, handle, *, force=False):
            return source

        def is_downloaded(self, handle):
            return False

    manager = manager_factory(downloader=FakeDownloader())
    path = manager.download()
    assert path == manager.settings.catalog_root
    assert (path / "2019_images" / "x.csv").exists()


def test_pipeline_end_to_end(synthetic_dataset: Path, manager_factory):
    manager = manager_factory(synthetic_dataset)
    inventory = manager.scan(use_cache=False)
    assert inventory.counts()["geotiff"] == 3

    written = manager.generate_metadata(force=True)
    assert written == 4
    assert manager.metadata_count() == 4

    report = manager.validate()
    assert report.passed is True

    summary = manager.summary()
    assert summary.total_files == 4
    assert summary.years_covered == [2019]

    # Versioning through the facade.
    assert manager.current_version() == "0.0.0"
    manager.bump_version("minor", message="baseline")
    assert manager.current_version() == "0.1.0"


def test_list_and_load(synthetic_dataset: Path, manager_factory):
    manager = manager_factory(synthetic_dataset)
    manager.generate_metadata(force=True)

    csvs = manager.list_csvs()
    assert len(csvs) == 1
    frame = manager.load_csv(csvs[0])
    assert list(frame.columns) == ["village", "crop", "yield_kg", "rainfall_mm"]

    images = manager.list_images(index_type=IndexType.NDVI, year=2019)
    assert len(images) == 2
    preview = manager.preview_csv(csvs[0], n_rows=2)
    assert len(preview) == 2

    # Metadata lookups.
    rec = manager.get_metadata(images[0])
    assert rec is not None and rec.index_type is IndexType.NDVI
    assert len(manager.query_metadata(year=2019)) == 4


def test_load_rejects_paths_outside_root(synthetic_dataset: Path, manager_factory, tmp_path: Path):
    manager = manager_factory(synthetic_dataset)
    outside = tmp_path / "secret.csv"
    outside.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(DatasetNotFoundError):
        manager.load_csv(outside)


def test_scan_empty_root_returns_empty_inventory(manager_factory, tmp_path: Path):
    # The manager creates its dataset root on construction, so a fresh root
    # scans as an empty (but valid) inventory.
    manager = manager_factory()
    inventory = manager.scan()
    assert inventory.counts()["total"] == 0


def test_load_rejects_unknown_file(synthetic_dataset: Path, manager_factory):
    manager = manager_factory(synthetic_dataset)
    with pytest.raises(DatasetNotFoundError):
        manager.load_csv("raw/kaggle-crop-yield/does-not-exist.csv")


def test_cache_via_facade(synthetic_dataset: Path, manager_factory):
    manager = manager_factory(synthetic_dataset)
    manager.cache_set("probe", {"x": 1}, ttl_seconds=60)
    assert manager.cache_get("probe") == {"x": 1}
    assert manager.cache_invalidate("probe") == 1
    assert manager.cache_get("probe") is None


def test_registry_and_versions_via_facade(synthetic_dataset: Path, manager_factory):
    manager = manager_factory(synthetic_dataset)
    entries = manager.registry_entries()
    assert len(entries) == 1 and entries[0]["name"] == "kaggle-crop-yield"

    manager.bump_version("patch", message="one")
    manager.bump_version("patch", message="two")
    versions = manager.list_versions()
    assert [v.version for v in versions] == ["0.0.2", "0.0.1"]
    manager.rollback_version("0.0.1")
    assert manager.current_version() == "0.0.1"


def test_context_manager(synthetic_dataset: Path, manager_factory):
    with manager_factory(synthetic_dataset) as manager:
        assert manager.scan().counts()["csv"] == 1


def test_info(synthetic_dataset: Path, manager_factory):
    manager = manager_factory(synthetic_dataset)
    info = manager.info()
    assert info["kaggle_handle"].startswith("shathanandabhatn")
    assert info["dependencies"]["pandas"] is not None
