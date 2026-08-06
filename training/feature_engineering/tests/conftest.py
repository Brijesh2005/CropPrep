"""Shared fixtures for the feature-engineering test-suite.

Reuses the STAM synthetic dataset (real Dataset Manager + GeoTIFFs + boundary
GeoJSON) and adds a resolved R2.3 :class:`ObservationCorpus` fixture so tests
exercise feature builders / statistics / balancing over real observations.
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
def manager(synthetic_catalog: Path):
    from training.dataset_manager import DatasetManager, Settings

    dataset_root = synthetic_catalog.parent.parent
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
    from training.stam import STAM

    instance = STAM(manager, stam_config)
    instance.initialize()
    return instance


@pytest.fixture
def corpus(stam):
    """A small resolved corpus: 2020+2021 Kharif over up to 2 locations."""
    from training.stam import ObservationResolver, ObservationResolverConfig

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
def accepted(corpus):
    """Accepted observations from the fixture corpus."""
    return corpus.accepted_observations()
