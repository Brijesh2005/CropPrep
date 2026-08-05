"""Shared fixtures for the STAM test-suite.

Tests use a real Dataset Manager pointed at a synthetic dataset tree (tabular
CSV + NDVI/EVI GeoTIFFs + admin-boundary GeoJSON) so the full
DatasetManager -> STAM integration is exercised without touching real data or
the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from training.dataset_manager.tests.helpers import make_tiff

# Make the repository root importable regardless of where pytest runs from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_synthetic_dataset(root: Path) -> Path:
    """Create the Kaggle-style tree under ``root`` and return the catalog path.

    Layout::

        <root>/datasets/raw/kaggle-crop-yield/
        ├── 2019_images/R10m/    (NDVI + EVI on 2019-07-01)
        ├── 2020_images/R10m/    (NDVI + EVI on 3 Kharif dates)
        ├── 2021_images/R10m/    (NDVI only on one date — tests missing EVI)
        ├── crop_yield.csv
        └── boundaries.geojson
    """
    datasets = root / "datasets"
    catalog = datasets / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    (catalog / "2020_images" / "R10m").mkdir(parents=True)
    (catalog / "2021_images" / "R10m").mkdir(parents=True)

    # Tabular record table.
    pd.DataFrame(
        {
            "village": ["A", "B", "A"],
            "district": ["DK", "DK", "DK"],
            "crop": ["Rice", "Rice", "Coconut"],
            "yield_kg": [5200, 5400, 3100],
            "year": [2020, 2020, 2021],
            "season": ["Kharif", "Kharif", "Kharif"],
            "rainfall_mm": [2100, 2050, 2300],
        }
    ).to_csv(catalog / "crop_yield.csv", index=False)

    # GeoTIFFs. Top-left origin (74.80, 13.10), pixel 0.0001 deg -> 40x40 px
    # footprint spanning lon 74.800..74.804, lat 13.096..13.100. EPSG:4326
    # keeps the degree-based transform consistent with the WGS-84 query
    # points (no projection needed in tests). Village-A polygon
    # (74.79..74.81, 13.08..13.11) contains this footprint.
    def _tiff(subdir: str, name: str, seed: int, day: str) -> Path:
        return make_tiff(
            catalog / subdir / f"{name}_{day}.tif",
            width=40, height=40, seed=seed, fill=None, crs="EPSG:4326",
            origin=(74.80, 13.10),
        )

    _tiff("2019_images/R10m", "S2_NDVI", 1, "20190701")
    _tiff("2019_images/R10m", "S2_EVI", 11, "20190701")

    for day, seed in [("20200601", 2), ("20200715", 3), ("20200901", 4)]:
        _tiff("2020_images/R10m", "S2_NDVI", seed, day)
        _tiff("2020_images/R10m", "S2_EVI", seed + 10, day)

    _tiff("2021_images/R10m", "S2_NDVI", 5, "20210701")  # missing EVI on purpose

    # Admin boundaries (village / taluk / district polygons in EPSG:4326).
    boundaries = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "A", "level": "village"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.79, 13.08], [74.81, 13.08],
                                     [74.81, 13.11], [74.79, 13.11], [74.79, 13.08]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "B", "level": "village"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.82, 13.08], [74.84, 13.08],
                                     [74.84, 13.11], [74.82, 13.11], [74.82, 13.08]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "T1", "level": "taluk"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.78, 13.07], [74.85, 13.07],
                                     [74.85, 13.12], [74.78, 13.12], [74.78, 13.07]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "DK", "level": "district"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.75, 13.05], [74.90, 13.05],
                                     [74.90, 13.20], [74.75, 13.20], [74.75, 13.05]]],
                },
            },
        ],
    }
    (catalog / "boundaries.geojson").write_text(
        json.dumps(boundaries), encoding="utf-8"
    )
    return catalog


@pytest.fixture
def synthetic_catalog(tmp_path: Path) -> Path:
    return _build_synthetic_dataset(tmp_path)


@pytest.fixture
def manager(synthetic_catalog: Path):
    """A real Dataset Manager over the synthetic dataset."""
    from training.dataset_manager import DatasetManager, Settings

    # synthetic_catalog = <tmp>/datasets/raw/kaggle-crop-yield
    dataset_root = synthetic_catalog.parent.parent  # <tmp>/datasets
    dm = DatasetManager(
        Settings(
            dataset_root=dataset_root,
            catalog_name="kaggle-crop-yield",
            logging={"console": False, "level": "ERROR"},
        )
    )
    dm.generate_metadata(force=True)
    return dm


@pytest.fixture
def stam_config(synthetic_catalog: Path):
    from training.stam import StamConfig

    return StamConfig(
        patch={"size": 16},
        tabular={"table": "crop_yield.csv",
                 "village_column": "village",
                 "district_column": "district",
                 "year_column": "year",
                 "season_column": "season",
                 "crop_column": "crop",
                 "yield_column": "yield_kg"},
        admin={"boundaries": ["raw/kaggle-crop-yield/boundaries.geojson"],
               "name_column": "name",
               "level_column": "level"},
        image={"resolution": "R10m", "require_pairs": True},
    )


@pytest.fixture
def stam(manager, stam_config):
    """An initialized STAM instance over the synthetic dataset."""
    from training.stam import STAM

    instance = STAM(manager, stam_config)
    instance.initialize()
    return instance
