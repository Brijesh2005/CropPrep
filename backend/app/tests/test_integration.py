"""End-to-end inference integration test over the real synthetic dataset.

Builds the app with the real Dataset Manager -> STAM -> preprocessing -> model
pipeline and exercises POST /predict and /explain.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app.core.config import load_settings
from app.main import create_app
from app.modules.inference.service import InferenceEngine
from app.services.model_registry import ModelRegistry


@pytest.fixture(scope="module")
def stam_chain():
    from services.dataset_manager import DatasetManager, Settings as DMSettings
    from services.spatial_alignment import STAM, StamConfig
    from services.spatial_alignment.tests.conftest import _build_synthetic_dataset

    tmp = Path(tempfile.mkdtemp())
    catalog = _build_synthetic_dataset(tmp)
    dataset_root = catalog.parent.parent
    manager = DatasetManager(
        DMSettings(
            dataset_root=dataset_root,
            catalog_name="kaggle-crop-yield",
            logging={"console": False, "level": "ERROR"},
        )
    )
    manager.generate_metadata(force=True)
    stam = STAM(
        manager,
        StamConfig(
            patch={"size": 16},
            tabular={"table": "crop_yield.csv", "village_column": "village",
                     "district_column": "district", "year_column": "year",
                     "season_column": "season", "crop_column": "crop",
                     "yield_column": "yield_kg"},
            admin={"boundaries": ["raw/kaggle-crop-yield/boundaries.geojson"],
                   "name_column": "name", "level_column": "level"},
            image={"resolution": "R10m", "require_pairs": True},
        ),
    )
    stam.initialize()
    return stam, dataset_root


@pytest.fixture(scope="module")
def fitted_preprocessor(stam_chain):
    from ai.preprocessing import PreprocessingConfig, Preprocessor

    stam, _ = stam_chain
    observations = []
    for lon, lat in [(74.801, 13.099), (74.802, 13.098), (74.803, 13.097)]:
        observations.append(stam.build_observation(lon, lat, year=2020, season="Kharif"))
    pre = Preprocessor(
        PreprocessingConfig(
            image={"size": 16, "normalize": "minmax"},
            temporal={"max_observations": 8, "min_observations": 1},
            tabular={"scaler": "standard", "categorical_encoding": "ordinal",
                     "numeric_features": ["rainfall_mm"],
                     "categorical_features": ["village", "district"],
                     "exclude_columns": ["crop", "yield_kg", "year", "season"]},
            quality={"min_quality_score": 0.0},
        )
    )
    accepted, _ = pre.filter(observations)
    pre.fit(accepted, extractor=stam.get_patch)
    return pre, observations


@pytest.fixture(scope="module")
def tiny_model(fitted_preprocessor):
    from ai.models import ModelFactory

    pre, _ = fitted_preprocessor
    return ModelFactory.create(
        ModelFactory.build_config(
            pre,
            temporal={"d_model": 32, "depth": 1, "num_heads": 2, "ff_dim": 128,
                      "embedding_dim": 32, "max_len": 8},
            shared_encoder={"d_model": 32, "depth": 1, "num_heads": 2, "ff_dim": 128,
                            "out_dim": 48},
        )
    ).eval()


@pytest.fixture(scope="module")
def integration_client(stam_chain, fitted_preprocessor, tiny_model, tmp_path_factory):
    stam, dataset_root = stam_chain
    pre, _ = fitted_preprocessor
    settings = load_settings(
        env={
            "BACKEND_DATABASE__URL": f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('db')}/int.db",
            "BACKEND_ENVIRONMENT": "test",
            "BACKEND_RATE_LIMIT__ENABLED": "false",
            "BACKEND_LOG__JSON_LOGS": "false",
            "BACKEND_MODEL__WARMUP": "false",
            "BACKEND_DATASET__VALIDATE_ON_STARTUP": "false",
            "BACKEND_DATASET__DATASET_ROOT": str(dataset_root),
        }
    )
    app = create_app(settings)
    container = app.state.container

    registry = ModelRegistry(settings.model)
    registry.model = tiny_model
    registry.version = "integration-1.0"
    registry.ready = True
    engine = InferenceEngine(
        registry, stam=stam, preprocessor=pre,
        config=settings.inference, cache=container.services.resolve("cache"),
    )
    container.model.override("stam", stam)
    container.model.override("preprocessor", pre)
    container.model.override("model_registry", registry)
    container.model.override("inference_engine", engine)

    with TestClient(app) as client:
        yield client


def _auth(integration_client) -> dict[str, str]:
    integration_client.post(
        "/api/v1/auth/register",
        json={"email": "integration@crop.io", "password": "secret123", "full_name": "Integration"},
    )
    r = integration_client.post(
        "/api/v1/auth/login",
        data={"username": "integration@crop.io", "password": "secret123"},
    )
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_integration_predict(integration_client):
    r = integration_client.post(
        "/api/v1/predict", json={"lon": 74.802, "lat": 13.098, "year": 2020, "season": "Kharif"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_crop"] in {"Rice", "Coconut"}
    assert body["confidence"] > 0
    assert body["model_version"] == "integration-1.0"
    assert body["prediction_id"] is not None


def test_integration_explain(integration_client):
    headers = _auth(integration_client)
    r = integration_client.post(
        "/api/v1/explain",
        json={"lon": 74.802, "lat": 13.098, "year": 2020, "season": "Kharif"},
        headers=headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["crop"]
    assert body["top_features"] is not None
