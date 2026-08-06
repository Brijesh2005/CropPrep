"""Shared pytest fixtures for the Dataset Manager test-suite.

Every test uses an isolated temporary dataset tree so the suite never touches
real project data or the user's Kaggle cache. Synthetic GeoTIFFs are created
via :func:`~services.dataset_manager.tests.helpers.make_tiff`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repository root importable regardless of where pytest runs from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import pytest

from training.dataset_manager.tests.helpers import make_tiff  # noqa: F401  (re-exported)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build a small but realistic Kaggle-style dataset tree.

    Layout::

        <tmp>/datasets/
        └── raw/kaggle-crop-yield/
            ├── 2019_images/
            │   ├── R10m/   (S2_NDVI_20190701.tif, S2_EVI_20190701.tif)
            │   └── NDVI/   (duplicate tiff, same name+size)
            └── crop_yield_2019.csv
    """
    root = tmp_path / "datasets"
    catalog = root / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    (catalog / "2019_images" / "NDVI").mkdir(parents=True)

    pd.DataFrame(
        {
            "village": ["Moodabidri", "Bantwal", "Sullia"],
            "crop": ["Rice", "Rice", "Coconut"],
            "yield_kg": [5200, 5400, 3100],
            "rainfall_mm": [2100, 2050, 2300],
        }
    ).to_csv(catalog / "crop_yield_2019.csv", index=False)

    make_tiff(catalog / "2019_images" / "R10m" / "S2_NDVI_20190701.tif", seed=1)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_EVI_20190701.tif", seed=2)
    # Duplicate (same name + size) inside NDVI/ to exercise duplicate checks.
    make_tiff(catalog / "2019_images" / "NDVI" / "S2_NDVI_20190701.tif", seed=1)
    return root


@pytest.fixture
def r22_dataset(tmp_path: Path) -> Path:
    """Multi-source R2.2 dataset: location-keyed CSV + multi-year NDVI/EVI.

    Layout::

        <tmp>/datasets/
        └── raw/kaggle-crop-yield/
            ├── crop_yield.csv            (village/district/lat/lon/year)
            ├── 2019_images/R10m/         (S2_NDVI, S2_EVI near Moodabidri)
            └── 2020_images/R10m/         (S2_NDVI near Bantwal)
    """
    root = tmp_path / "datasets"
    catalog = root / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    (catalog / "2020_images" / "R10m").mkdir(parents=True)

    pd.DataFrame(
        {
            "village": ["Moodabidri", "Bantwal", "Sullia"],
            "district": ["Dakshina Kannada"] * 3,
            "latitude": [13.08, 12.90, 12.56],
            "longitude": [74.89, 75.00, 75.35],
            "year": [2019, 2019, 2020],
            "yield_kg": [5200, 5400, 3100],
        }
    ).to_csv(catalog / "crop_yield.csv", index=False)

    # Moodabidri (13.08, 74.89) lands inside the 2019 rasters.
    make_tiff(
        catalog / "2019_images" / "R10m" / "S2_NDVI_20190701.tif",
        seed=1, origin=(74.8895, 13.0805), crs="EPSG:4326",
    )
    make_tiff(
        catalog / "2019_images" / "R10m" / "S2_EVI_20190701.tif",
        seed=2, origin=(74.8895, 13.0805), crs="EPSG:4326",
    )
    # Bantwal (12.90, 75.00) lands inside the 2020 raster.
    make_tiff(
        catalog / "2020_images" / "R10m" / "S2_NDVI_20200715.tif",
        seed=3, origin=(74.9995, 12.9005), crs="EPSG:4326",
    )
    return root


@pytest.fixture
def r22_manager_factory(r22_dataset: Path):
    """Factory building a fully-wired R2.2 manager on :func:`r22_dataset`."""
    from training.dataset_manager import DatasetManager, Settings

    def _build(**kwargs):
        settings_overrides = kwargs.pop("settings_overrides", {})
        providers = {
            "tabular": {"root": str(r22_dataset / "raw" / "kaggle-crop-yield")},
        }
        override = settings_overrides.pop("providers", None)
        if override:
            providers = _deep_merge(providers, override)
        settings = Settings(
            dataset_root=r22_dataset,
            catalog_name="kaggle-crop-yield",
            logging={"console": False, "level": "ERROR"},
            providers=providers,
            **settings_overrides,
        )
        return DatasetManager(settings, **kwargs)

    return _build


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@pytest.fixture
def manager_factory(tmp_path: Path):
    """Factory building a :class:`DatasetManager` on an isolated dataset root.

    Extra keyword arguments are forwarded to the manager constructor, e.g.::

        manager = manager_factory(root, downloader=FakeDownloader())
    """
    from training.dataset_manager import DatasetManager, Settings

    def _build(root: Path | None = None, **kwargs):
        dataset_root = root or (tmp_path / "datasets")
        settings_overrides = kwargs.pop("settings_overrides", {})
        settings = Settings(
            dataset_root=dataset_root,
            catalog_name="kaggle-crop-yield",
            logging={"console": False, "level": "ERROR"},
            **settings_overrides,
        )
        return DatasetManager(settings, **kwargs)

    return _build


@pytest.fixture
def fake_kaggle(tmp_path: Path):
    """A controllable fake kagglehub module + cache root.

    The fake download source is created eagerly so tests can materialise from
    it without first triggering a ``dataset_download`` call.
    """
    import types

    cache_root = tmp_path / "kagglehub" / "datasets"
    downloaded_root = tmp_path / "download_src"
    (downloaded_root / "files").mkdir(parents=True, exist_ok=True)
    (downloaded_root / "files" / "S2_NDVI_2020.tif").write_bytes(b"fake-tif-data")

    class FakeKaggleHub:
        def __init__(self):
            self.calls: list[dict] = []
            self.fail_next = False

        def dataset_download(self, handle: str, force_download: bool = False) -> str:
            self.calls.append({"handle": handle, "force": force_download})
            if self.fail_next:
                raise RuntimeError("network unavailable")
            return str(downloaded_root)

    fake = FakeKaggleHub()
    return types.SimpleNamespace(
        module=fake,
        cache_root=cache_root,
        downloaded_root=downloaded_root,
    )
