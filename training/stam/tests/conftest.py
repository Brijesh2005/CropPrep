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
        ├── crop_yield.csv       (narrow, village-level)
        ├── icrisat_wide.csv     (wide, district-level AREA/YIELD triples)
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

    # Wide-format ICRISAT-style table (district-level, one AREA/PRODUCTION/
    # YIELD triple per crop, derived dominant crop by largest planted area).
    pd.DataFrame(
        {
            "State Name": ["Karnataka", "Karnataka", "Karnataka"],
            "Dist Name": ["DK", "DK", "XYZ"],
            "Year": [2020, 2019, 2020],
            "RICE AREA (1000 ha)": [100.0, 110.0, 200.0],
            "RICE PRODUCTION (1000 tons)": [4000.0, 4400.0, 8000.0],
            "RICE YIELD (Kg per ha)": [4000.0, 4000.0, 4000.0],
            "COTTON AREA (1000 ha)": [50.0, 40.0, 60.0],
            "COTTON PRODUCTION (1000 tons)": [300.0, 240.0, 360.0],
            "COTTON YIELD (Kg per ha)": [6000.0, 6000.0, 6000.0],
        }
    ).to_csv(catalog / "icrisat_wide.csv", index=False)

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
def stam_config_multi_table(synthetic_catalog: Path):
    """Two-table chain: village-level crop_yield.csv then district-level
    icrisat_wide.csv (mirrors the real data_season -> ICRISAT fallback)."""
    from training.stam import StamConfig

    return StamConfig(
        patch={"size": 16},
        tabular={
            "tables": [
                {"name": "crop_yield.csv",
                 "village_column": "village",
                 "district_column": "district",
                 "year_column": "year",
                 "season_column": "season",
                 "crop_column": "crop",
                 "yield_column": "yield_kg",
                 "fallback_to_district": False},
                {"name": "icrisat_wide.csv",
                 "district_column": "Dist Name",
                 "year_column": "Year",
                 "state_column": "State Name",
                 "state_value": "Karnataka",
                 "fallback_to_district": True},
            ]
        },
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


# --------------------------------------------------------------------------- #
# District/place-name alias fixtures (boundary spelling -> data_season Location)
# --------------------------------------------------------------------------- #


def _build_alias_dataset(root: Path) -> Path:
    """Dataset tree for the district-alias tests.

    Mirrors the real layout: ``tabular/data_season.csv`` (colloquial
    ``Location`` vocabulary, no district column) + ``raw/crop-alias/``
    boundaries whose district names use the KGIS spellings (``Dakshina
    Kannada``, ``Kalaburgi``, ...). No satellite imagery is included — the
    tabular match is the focus and an empty sequence is fine.
    """
    datasets = root / "datasets"
    catalog = datasets / "raw" / "crop-alias"
    catalog.mkdir(parents=True)
    (datasets / "tabular").mkdir(parents=True)

    # data_season.csv: same columns as the real table; one row per location
    # for 2018 (Mangalore also 2019). Belgaum deliberately absent.
    pd.DataFrame(
        {
            "Year": [2018, 2018, 2018, 2018, 2018, 2019],
            "Location": ["Mangalore", "Gulbarga", "Bangalore", "Madikeri",
                         "Kasaragodu", "Mangalore"],
            "Area": [52119, 22000, 15000, 8000, 10000, 53000],
            "Rainfall": [2903.1, 850.0, 900.0, 2400.0, 3000.0, 2910.0],
            "Temperature": [27, 29, 27, 22, 27, 27],
            "Soil type": ["Alluvial", "Black", "Red", "Laterite", "Alluvial",
                          "Alluvial"],
            "Irrigation": ["Drip", "Canal", "Tank", "Rainfed", "Drip", "Drip"],
            "yeilds": [114744, 86000, 54000, 32000, 41000, 116000],
            "Humidity": [57, 45, 52, 60, 58, 57],
            "Crops": ["Coconut", "Tur", "Ragi", "Coffee", "Coconut", "Coconut"],
            "price": [51239, 40000, 35000, 60000, 50000, 51500],
            "Season": ["Kharif", "Kharif", "Kharif", "Kharif", "Kharif", "Kharif"],
        }
    ).to_csv(datasets / "tabular" / "data_season.csv", index=False)

    # KGIS-style district polygons (official spellings) + a Madikeri taluk
    # nested inside Kodagu. Query points land on the centroid of the polygon
    # they live in (which is also the nearest location point).
    boundaries = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Dakshina Kannada", "level": "district"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.7, 12.5], [75.3, 12.5],
                                     [75.3, 13.3], [74.7, 13.3], [74.7, 12.5]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Kodagu", "level": "district"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[75.3, 12.0], [76.2, 12.0],
                                     [76.2, 12.6], [75.3, 12.6], [75.3, 12.0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Madikeri", "level": "taluk"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[75.3, 12.1], [75.8, 12.1],
                                     [75.8, 12.4], [75.3, 12.4], [75.3, 12.1]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Kalaburgi", "level": "district"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[76.5, 16.5], [77.4, 16.5],
                                     [77.4, 17.5], [76.5, 17.5], [76.5, 16.5]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Belgaum", "level": "district"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.2, 15.5], [75.1, 15.5],
                                     [75.1, 16.5], [74.2, 16.5], [74.2, 15.5]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Bengaluru (Urban)", "level": "district"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[77.3, 12.8], [77.8, 12.8],
                                     [77.8, 13.2], [77.3, 13.2], [77.3, 12.8]]],
                },
            },
        ],
    }
    (catalog / "boundaries.geojson").write_text(
        json.dumps(boundaries), encoding="utf-8"
    )
    return catalog


@pytest.fixture
def alias_catalog(tmp_path: Path) -> Path:
    return _build_alias_dataset(tmp_path)


@pytest.fixture
def alias_manager(alias_catalog: Path):
    """A real Dataset Manager over the district-alias dataset tree."""
    from training.dataset_manager import DatasetManager, Settings

    dataset_root = alias_catalog.parent.parent  # <tmp>/datasets
    dm = DatasetManager(
        Settings(
            dataset_root=dataset_root,
            catalog_name="crop-alias",
            logging={"console": False, "level": "ERROR"},
        )
    )
    dm.generate_metadata(force=True)
    return dm


@pytest.fixture
def alias_stam_config(alias_catalog: Path):
    """Single-table config: data_season.csv only (no district column)."""
    from training.stam import StamConfig

    return StamConfig(
        patch={"size": 16},
        tabular={
            "tables": [
                {"name": "data_season.csv",
                 "village_column": "Location",
                 "year_column": "Year",
                 "season_column": "Season",
                 "crop_column": "Crops",
                 "yield_column": "yeilds",
                 "fallback_to_district": False},
            ]
        },
        admin={"boundaries": ["raw/crop-alias/boundaries.geojson"],
               "name_column": "name",
               "level_column": "level"},
        image={"resolution": "R10m", "require_pairs": True},
    )


@pytest.fixture
def alias_stam(alias_manager, alias_stam_config):
    """An initialized STAM instance over the district-alias dataset."""
    from training.stam import STAM

    instance = STAM(alias_manager, alias_stam_config)
    instance.initialize()
    return instance
