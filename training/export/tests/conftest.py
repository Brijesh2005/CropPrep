"""Shared fixtures for the export test-suite.

Builds a small resolved corpus over the STAM synthetic dataset and a matching
feature frame via :func:`build_feature_frame`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from training.stam.tests.conftest import _build_synthetic_dataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def synthetic_catalog(tmp_path: Path) -> Path:
    return _build_synthetic_dataset(tmp_path)


@pytest.fixture
def corpus(synthetic_catalog: Path):
    from training.dataset_manager import DatasetManager, Settings
    from training.stam import ObservationResolver, ObservationResolverConfig, STAM, StamConfig

    dataset_root = synthetic_catalog.parent.parent
    dm = DatasetManager(
        Settings(
            dataset_root=dataset_root,
            catalog_name="kaggle-crop-yield",
            logging={"console": False, "level": "ERROR"},
        )
    )
    dm.generate_metadata(force=True)
    stam = STAM(
        dm,
        StamConfig(
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
        ),
    )
    stam.initialize()

    resolver = ObservationResolver(
        stam,
        ObservationResolverConfig(
            min_quality_score=0.0,
            include_rejected=True,
            include_errors=True,
            use_cache=False,
        ),
    )
    plan = resolver.plan(years=[2020, 2021], seasons=["Kharif"], max_locations=2)
    return resolver.resolve(plan)


@pytest.fixture
def frame(corpus):
    from training.feature_engineering import build_feature_frame

    frame = build_feature_frame(corpus)
    if not len(frame):
        pytest.skip("no accepted observations in fixture corpus")
    return frame
