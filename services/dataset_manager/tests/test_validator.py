"""Tests for the dataset validator."""

from __future__ import annotations

from pathlib import Path

from services.dataset_manager.config import ValidateConfig
from services.dataset_manager.image_loader import RasterioImageLoader
from services.dataset_manager.scanner import DatasetScanner
from services.dataset_manager.tests.helpers import make_tiff
from services.dataset_manager.validator import DatasetValidator


def _validate(root: Path, **config_overrides):
    scanner = DatasetScanner()
    inventory = scanner.scan(root, use_cache=False)
    validator = DatasetValidator(
        ValidateConfig(**config_overrides), image_loader=RasterioImageLoader()
    )
    return validator.validate(root, inventory)


def test_healthy_tree_passes(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    report = _validate(root)
    assert report.passed is True
    assert report.files_scanned == 4


def test_empty_csv_is_error(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    (root / "empty.csv").write_bytes(b"")
    report = _validate(root)
    assert report.passed is False
    assert any(i.code == "V-CSV-001" for i in report.issues)


def test_corrupted_tiff_is_error(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    (root / "bad.tif").write_bytes(b"garbage")
    report = _validate(root)
    assert report.passed is False
    assert any(i.code in {"V-RAST-001", "V-RAST-002"} for i in report.issues)


def test_duplicate_files_flagged(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    report = _validate(root)
    assert any(i.code == "V-DUP-001" for i in report.issues)


def test_year_gap_reports_warning(synthetic_dataset: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    report = _validate(root)
    assert any(i.code == "V-STRUCT-004" for i in report.issues)


def test_no_crs_is_warning(tmp_path: Path):
    root = tmp_path / "catalog"
    make_tiff(root / "NDVI" / "S2_NDVI_2020.tif", crs=None)
    report = _validate(root)
    assert any(i.code == "V-RAST-003" for i in report.issues)
    # Missing CRS is a warning, so the report still passes.
    assert report.passed is True


def test_fail_on_warning_promotes_crs(tmp_path: Path):
    root = tmp_path / "catalog"
    make_tiff(root / "NDVI" / "S2_NDVI_2020.tif", crs=None)
    report = _validate(root, fail_on_warning=True)
    assert report.passed is False


def test_missing_metadata_flagged(synthetic_dataset: Path, tmp_path: Path):
    from services.dataset_manager.metadata import SQLiteMetadataStore

    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    store = SQLiteMetadataStore(tmp_path / "meta.db")
    # No records generated -> every file is missing metadata.
    report = DatasetValidator(
        ValidateConfig(require_metadata=True),
        image_loader=RasterioImageLoader(),
        metadata_store=store,
    ).validate(root, DatasetScanner().scan(root, use_cache=False))
    assert any(i.code == "V-META-003" for i in report.issues)


def test_orphaned_metadata_flagged(synthetic_dataset: Path, tmp_path: Path):
    from services.dataset_manager.csv_loader import PandasCSVLoader
    from services.dataset_manager.metadata import MetadataGeneratorImpl, SQLiteMetadataStore

    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    store = SQLiteMetadataStore(tmp_path / "meta.db")
    scanner = DatasetScanner()
    inventory = scanner.scan(root, use_cache=False)
    generator = MetadataGeneratorImpl(
        csv_loader=PandasCSVLoader(),
        image_loader=RasterioImageLoader(),
        store=store,
    )
    generator.generate(root, inventory)

    # Remove one file on disk but keep its metadata record.
    victim = next(e.path for e in inventory.entries if e.extension == "csv")
    victim.unlink()

    report = DatasetValidator(
        ValidateConfig(require_metadata=True),
        image_loader=RasterioImageLoader(),
        metadata_store=store,
    ).validate(root, scanner.scan(root, use_cache=False))
    assert any(i.code == "V-META-002" for i in report.issues)
