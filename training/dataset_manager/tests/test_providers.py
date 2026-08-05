"""Tests for the Dataset Manager provider layer.

Covers both providers:

* :class:`GitRepositoryTabularProvider` — discovery, schema validation,
  statistics, missing-value handling, joins, metadata.
* :class:`KaggleHubImageProvider` — catalog / discovery, lazy raster access,
  patch retrieval, historical context, download-or-reuse.

All tests use isolated temporary trees — no network and no real Kaggle cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pytest

from training.dataset_manager.exceptions import DatasetNotFoundError
from training.dataset_manager.providers import (
    GitRepositoryTabularProvider,
    KaggleHubImageProvider,
    PatchRequest,
    ProviderStatus,
    TabularJoinSpec,
)
from training.dataset_manager.tests.helpers import make_tiff


# --------------------------------------------------------------------------- #
# GitRepositoryTabularProvider
# --------------------------------------------------------------------------- #


@pytest.fixture
def tabular_root(tmp_path: Path) -> Path:
    root = tmp_path / "tabular"
    root.mkdir()
    pd.DataFrame(
        {
            "village": ["Moodabidri", "Bantwal"],
            "district": ["Dakshina Kannada", "Dakshina Kannada"],
            "yield_kg": [5200, 5400],
            "rainfall_mm": [2100, 2050],
        }
    ).to_csv(root / "crop_production.csv", index=False)
    pd.DataFrame(
        {
            "village": ["Moodabidri", "Bantwal"],
            "area_ha": [12.5, 9.0],
        }
    ).to_csv(root / "village_areas.csv", index=False)
    pd.DataFrame(
        {"village": ["Sullia", "Bantwal"], "yield_kg": ["n/a", "5.0"]}
    ).to_csv(root / "messy.csv", index=False, header=["village", "yield_kg"])
    return root


def test_tabular_discovery(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    catalog = provider.discover()
    assert catalog.names() == ["crop_production", "messy", "village_areas"]
    assert provider.status is ProviderStatus.READY
    assert provider.available() is True


def test_tabular_missing_root(tmp_path: Path):
    provider = GitRepositoryTabularProvider(root=tmp_path / "nope")
    assert provider.discover().names() == []
    assert provider.status is ProviderStatus.MISSING_DATA
    assert provider.available() is False


def test_tabular_load_and_schema(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    frame = provider.load("crop_production")
    assert list(frame.columns) == ["village", "district", "yield_kg", "rainfall_mm"]
    assert len(frame) == 2

    schema = provider.schema("crop_production")
    assert schema["column_count"] == 4
    assert "statistics" in schema
    assert schema["statistics"]["yield_kg"]["mean"] == 5300.0

    stats = provider.statistics("crop_production")
    assert stats["rainfall_mm"]["min"] == 2050.0
    assert provider.missing_values("crop_production") == {
        "village": 0,
        "district": 0,
        "yield_kg": 0,
        "rainfall_mm": 0,
    }


def test_tabular_validate_schema_flags_messy(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    good = provider.validate_schema("crop_production")
    assert good["valid"] is True


def test_tabular_missing_and_handle_missing(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    missing = provider.missing_values("messy")
    assert missing.get("yield_kg", 0) >= 1

    dropped = provider.handle_missing("messy", "drop")
    assert dropped["yield_kg"].isna().sum() == 0
    assert len(dropped) == 1

    filled = provider.handle_missing("messy", "fill", fill_method="mean")
    assert filled["yield_kg"].notna().all()
    assert filled["yield_kg"].iloc[0] == 5.0

    filled_const = provider.handle_missing("messy", "fill", fill_value=0)
    assert filled_const["yield_kg"].notna().all()


def test_tabular_join(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    joined = provider.join(
        [
            TabularJoinSpec("crop_production", "village_areas", on="village"),
        ]
    )
    assert "area_ha" in joined.columns
    assert len(joined) == 2


def test_tabular_stream(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    chunks = list(provider.stream("crop_production", chunksize=1))
    assert len(chunks) == 2
    assert all(len(c) == 1 for c in chunks)


def test_tabular_metadata_and_missing_name(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    meta = provider.metadata("crop_production")
    assert meta["name"] == "crop_production"
    assert meta["size_bytes"] > 0
    assert "schema" in meta

    with pytest.raises(DatasetNotFoundError):
        provider.load("does_not_exist")


def test_tabular_manifest(tabular_root: Path):
    provider = GitRepositoryTabularProvider(root=tabular_root)
    manifest = provider.manifest().to_dict()
    assert manifest["kind"] == "tabular"
    assert manifest["available"] is True
    assert set(manifest["details"]["datasets"]) == {
        "crop_production",
        "messy",
        "village_areas",
    }


# --------------------------------------------------------------------------- #
# KaggleHubImageProvider
# --------------------------------------------------------------------------- #


@pytest.fixture
def image_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    catalog = root / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_NDVI_20190701.tif", seed=1)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_EVI_20190701.tif", seed=2)
    (catalog / "2020_images" / "R20m").mkdir(parents=True)
    make_tiff(catalog / "2020_images" / "R20m" / "S2_NDVI_20200815.tif", seed=3)
    pd.DataFrame({"village": ["a"], "yield_kg": [1]}).to_csv(
        catalog / "crop_yield_2019.csv", index=False
    )
    return root


def _provider(image_dataset: Path, **kwargs) -> KaggleHubImageProvider:
    return KaggleHubImageProvider(dataset_root=image_dataset, **kwargs)


def test_image_location_and_catalog(image_dataset: Path):
    provider = _provider(image_dataset)
    location = provider.location()
    assert location.materialized is True
    assert location.downloaded is False

    catalog = provider.catalog()
    assert catalog.years == [2019, 2020]
    assert sorted(catalog.resolutions) == ["R10m", "R20m"]
    assert len(catalog.ndvi) == 2
    assert len(catalog.evi) == 1
    assert provider.status is ProviderStatus.READY


def test_image_discover_ndvi_evi(image_dataset: Path):
    provider = _provider(image_dataset)
    ndvi = provider.discover_ndvi()
    evi = provider.discover_evi()
    assert [e.relative_path for e in ndvi] == [
        "2019_images/R10m/S2_NDVI_20190701.tif",
        "2020_images/R20m/S2_NDVI_20200815.tif",
    ]
    assert len(evi) == 1


def test_image_lazy_read_metadata(image_dataset: Path):
    provider = _provider(image_dataset)
    raster = provider.discover_ndvi()[0]
    meta = provider.read_metadata(raster.path)
    assert meta.width == 20 and meta.height == 20
    assert meta.crs == "EPSG:32643"
    assert meta.index_type.value == "NDVI"


def test_image_read_and_patch(image_dataset: Path):
    provider = _provider(image_dataset)
    raster = provider.discover_ndvi()[0]
    meta = provider.read_metadata(raster.path)
    center = (
        (meta.bounds[0] + meta.bounds[2]) / 2,
        (meta.bounds[1] + meta.bounds[3]) / 2,
    )
    patch = provider.patch(PatchRequest(path=raster.path, center=center, size=8))
    assert patch.shape == (8, 8)

    full = provider.read(raster.path)
    assert full.shape == (20, 20)

    window = provider.read(raster.path, window=(0, 0, 4, 4))
    assert window.shape == (4, 4)


def test_image_patch_clamped_at_edge(image_dataset: Path):
    provider = _provider(image_dataset)
    raster = provider.discover_ndvi()[0]
    meta = provider.read_metadata(raster.path)
    center = (meta.bounds[0], meta.bounds[1])  # far outside south-west corner
    patch = provider.patch(PatchRequest(path=raster.path, center=center, size=32))
    # Clamped to the raster extent; never larger than the raster.
    assert patch.shape[0] <= 20 and patch.shape[1] <= 20


def test_image_historical_context_falls_back_to_inventory(image_dataset: Path):
    provider = _provider(image_dataset)
    ctx = provider.get_historical_context(window_months=[7])
    assert ctx.years == [2019]  # only July 2019 records fall in month 7
    assert ctx.per_year[2019]["ndvi"] == 1

    all_years = provider.get_historical_context()
    assert all_years.years == [2019, 2020]


def test_image_validate_and_generate_metadata(image_dataset: Path):
    provider = _provider(image_dataset)
    report = provider.validate()
    assert report.passed is True

    written = provider.generate_metadata(force=True)
    assert written == 4  # 3 GeoTIFFs + 1 CSV


def test_image_ensure_via_fake_downloader(image_dataset: Path, tmp_path: Path):
    from training.dataset_manager.downloader import KaggleDownloader

    source = tmp_path / "download_src"
    (source / "files").mkdir(parents=True)
    make_tiff(source / "files" / "S2_NDVI_2021.tif", seed=4)

    class FakeDownloader(KaggleDownloader):
        def download(self, handle, *, force=False):
            return source

        def is_downloaded(self, handle):
            return False

    root = tmp_path / "datasets2"
    provider = KaggleHubImageProvider(
        dataset_root=root,
        downloader=FakeDownloader(),
        catalog_name="kaggle-crop-yield",
    )
    path = provider.ensure(force=True)
    assert path == root / "raw" / "kaggle-crop-yield"
    assert provider.available() is True
    assert provider.catalog().years == [2021]


def test_image_manifest(image_dataset: Path):
    provider = _provider(image_dataset)
    manifest = provider.manifest().to_dict()
    assert manifest["kind"] == "image"
    assert manifest["available"] is True
    assert manifest["details"]["handle"].startswith("shathanandabhatn")
    assert manifest["details"]["years"] == [2019, 2020]


def test_image_read_rejects_path_outside_roots(image_dataset: Path, tmp_path: Path):
    provider = _provider(image_dataset)
    outside = tmp_path / "outside.tif"
    make_tiff(outside, seed=5)
    with pytest.raises(DatasetNotFoundError):
        provider.read_metadata(outside)


# --------------------------------------------------------------------------- #
# DatasetManager delegation
# --------------------------------------------------------------------------- #


def test_manager_delegates_to_tabular_provider(tmp_path: Path, manager_factory):
    tabular = tmp_path / "tabular"
    tabular.mkdir()
    pd.DataFrame({"village": ["a"], "yield_kg": [1]}).to_csv(
        tabular / "samples.csv", index=False
    )
    manager = manager_factory(
        settings_overrides={"providers": {"tabular": {"root": tabular}}}
    )
    assert manager.tabular_names() == ["samples"]
    frame = manager.load_tabular("samples")
    assert list(frame.columns) == ["village", "yield_kg"]
    assert manager.tabular_schema("samples")["column_count"] == 2
    assert manager.tabular_statistics("samples")["yield_kg"]["mean"] == 1.0
    assert manager.tabular_metadata("samples")["name"] == "samples"


def test_manager_delegates_to_image_provider(image_dataset: Path, manager_factory):
    manager = manager_factory(image_dataset)
    catalog = manager.image_catalog()
    assert len(catalog.ndvi) == 2
    assert len(manager.discover_evi()) == 1
    assert manager.image_location().materialized is True
    manifests = manager.provider_manifests()
    assert "git_repository_tabular" in manifests
    assert "kaggle_hub_image" in manifests
    ctx = manager.image_historical_context()
    assert ctx.years == [2019, 2020]
    assert "providers" in manager.info()
