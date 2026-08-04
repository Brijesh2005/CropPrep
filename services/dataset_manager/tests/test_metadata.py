"""Tests for metadata generation and the SQLite metadata store."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.dataset_manager.csv_loader import PandasCSVLoader
from services.dataset_manager.exceptions import InvalidMetadataError
from services.dataset_manager.image_loader import RasterioImageLoader
from services.dataset_manager.metadata import MetadataGeneratorImpl, SQLiteMetadataStore
from services.dataset_manager.models import IndexType
from services.dataset_manager.scanner import DatasetScanner


def _generate(root: Path, db_path: Path, **config):
    from services.dataset_manager.config import MetadataConfig

    store = SQLiteMetadataStore(db_path)
    generator = MetadataGeneratorImpl(
        MetadataConfig(**config),
        csv_loader=PandasCSVLoader(),
        image_loader=RasterioImageLoader(),
        store=store,
    )
    inventory = DatasetScanner().scan(root, use_cache=False)
    records = generator.generate(root, inventory)
    return records, store


def test_generate_metadata_for_all_files(synthetic_dataset: Path, tmp_path: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    records, store = _generate(root, tmp_path / "meta.db")
    assert len(records) == 4
    assert store.count() == 4


def test_generate_is_idempotent(synthetic_dataset: Path, tmp_path: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    _generate(root, tmp_path / "meta.db")
    # Second run produces nothing new (file sizes unchanged).
    records2, _ = _generate(root, tmp_path / "meta.db")
    assert records2 == []


def test_csv_record_has_schema(synthetic_dataset: Path, tmp_path: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    records, _ = _generate(root, tmp_path / "meta.db")
    csv_record = next(r for r in records if r.category.value == "csv")
    assert csv_record.row_count == 3
    assert csv_record.column_count == 4
    assert "village" in csv_record.columns_json
    assert csv_record.sha256 is not None


def test_raster_record_has_geospatial_fields(synthetic_dataset: Path, tmp_path: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    records, _ = _generate(root, tmp_path / "meta.db")
    raster = next(r for r in records if r.category.value == "geotiff")
    assert raster.index_type in {IndexType.NDVI, IndexType.EVI}
    assert raster.width == 20 and raster.height == 20
    assert raster.crs == "EPSG:32643"
    assert raster.pixel_size is not None
    assert raster.bounds is not None
    assert raster.sha256 is not None


def test_compute_hashes_can_be_disabled(synthetic_dataset: Path, tmp_path: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    records, _ = _generate(root, tmp_path / "meta.db", compute_hashes=False)
    assert all(r.sha256 is None for r in records)


def test_store_get_and_query_roundtrip(synthetic_dataset: Path, tmp_path: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    _, store = _generate(root, tmp_path / "meta.db")

    # Query by filters.
    ndvi_records = store.query(index_type=IndexType.NDVI)
    assert len(ndvi_records) == 2
    year_records = store.query(year=2019)
    assert len(year_records) == 4

    # Get by path.
    first = store.query(index_type="NDVI")[0]
    fetched = store.get(Path(first.path))
    assert fetched is not None and fetched.relative_path == first.relative_path


def test_export_parquet(synthetic_dataset: Path, tmp_path: Path):
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    _, store = _generate(root, tmp_path / "meta.db")
    out = tmp_path / "metadata.parquet"
    try:
        store.export_parquet(out)
    except InvalidMetadataError as exc:  # pyarrow/fastparquet missing
        pytest.skip(f"parquet engine unavailable: {exc}")
    assert out.exists() and out.stat().st_size > 0


def test_save_many_replaces_existing(synthetic_dataset: Path, tmp_path: Path):
    from services.dataset_manager.models import MetadataRecord

    store = SQLiteMetadataStore(tmp_path / "meta.db")
    root = synthetic_dataset / "raw" / "kaggle-crop-yield"
    records, _ = _generate(root, tmp_path / "meta.db")
    store.save_many(records)
    assert store.count() == 4
    # Re-saving the same relative paths replaces (upsert) without duplication.
    store.save_many(records)
    assert store.count() == 4


def test_unexpected_query_filter_rejected(synthetic_dataset: Path, tmp_path: Path):
    from services.dataset_manager.metadata import SQLiteMetadataStore

    store = SQLiteMetadataStore(tmp_path / "meta.db")
    with pytest.raises(InvalidMetadataError):
        store.query(bogus_filter=1)
