"""Shared fixtures for the preprocessing pipeline tests.

Observations are built by running a real STAM instance over the synthetic
dataset (same fixtures the STAM suite uses), so the full
DatasetManager -> STAM -> Preprocessor chain is exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from services.spatial_alignment.tests.conftest import _build_synthetic_dataset

# Make the repository root importable regardless of where pytest runs from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def stam_instance(tmp_path: Path):
    """A real, initialized STAM over the synthetic dataset."""
    from services.dataset_manager import DatasetManager, Settings
    from services.spatial_alignment import STAM, StamConfig

    catalog = _build_synthetic_dataset(tmp_path)
    dataset_root = catalog.parent.parent
    manager = DatasetManager(
        Settings(
            dataset_root=dataset_root,
            catalog_name="kaggle-crop-yield",
            logging={"console": False, "level": "ERROR"},
        )
    )
    manager.generate_metadata(force=True)

    stam = STAM(
        manager,
        StamConfig(
            patch={"size": 32},
            tabular={"table": "crop_yield.csv",
                     "village_column": "village",
                     "district_column": "district",
                     "year_column": "year",
                     "season_column": "season",
                     "crop_column": "crop",
                     "yield_column": "yield_kg"},
            admin={"boundaries": ["raw/kaggle-crop-yield/boundaries.geojson"],
                   "name_column": "name", "level_column": "level"},
            image={"resolution": "R10m", "require_pairs": True},
        ),
    )
    stam.initialize()
    return stam


@pytest.fixture
def observations(stam_instance):
    """A set of observations across points and years (2020 Kharif ideal)."""
    obs = []
    for lon, lat in [(74.801, 13.099), (74.802, 13.098), (74.803, 13.097)]:
        obs.append(stam_instance.build_observation(lon, lat, year=2020, season="Kharif"))
    obs.append(stam_instance.build_observation(74.802, 13.098, year=2021, season="Kharif"))
    return obs


@pytest.fixture
def extractor(stam_instance):
    """Patch extractor: STAM.get_patch bound to the observation location."""
    return stam_instance.get_patch


@pytest.fixture
def preprocessing_config():
    from ai.preprocessing import PreprocessingConfig

    return PreprocessingConfig(
        image={"size": 32, "normalize": "minmax"},
        temporal={"max_observations": 8, "min_observations": 1},
        tabular={
            "scaler": "standard",
            "categorical_encoding": "onehot",
            "numeric_features": ["rainfall_mm"],
            "categorical_features": ["village", "district"],
            "exclude_columns": ["crop", "yield_kg", "year", "season"],
        },
        split={"strategy": "temporal", "test_years": [2021], "val_years": []},
        quality={"min_quality_score": 0.0},
    )
