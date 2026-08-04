"""Shared fixtures for the backend test-suite.

Most tests use an in-memory SQLite database and stub the AI/model container so
no real model or dataset is needed. An integration fixture builds the real
STAM -> preprocessing -> model pipeline over the synthetic dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture
def settings(tmp_path) -> "object":
    from app.core.config import load_settings

    return load_settings(
        env={
            "BACKEND_DATABASE__URL": f"sqlite+aiosqlite:///{tmp_path}/test.db",
            "BACKEND_ENVIRONMENT": "test",
            "BACKEND_RATE_LIMIT__ENABLED": "false",
            "BACKEND_LOG__JSON_LOGS": "false",
            "BACKEND_MODEL__WARMUP": "false",
            "BACKEND_DATASET__VALIDATE_ON_STARTUP": "false",
        }
    )


@pytest.fixture
def app(settings):
    from app.main import create_app

    return create_app(settings)


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    client.post(
        "/api/v1/auth/register",
        json={"email": "tester@crop.io", "password": "secret123", "full_name": "Tester"},
    )
    r = client.post(
        "/api/v1/auth/login",
        data={"username": "tester@crop.io", "password": "secret123"},
    )
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(client) -> dict[str, str]:
    client.post(
        "/api/v1/auth/register",
        json={"email": "admin@crop.io", "password": "secret123", "full_name": "Admin"},
    )
    r = client.post(
        "/api/v1/auth/login",
        data={"username": "admin@crop.io", "password": "secret123"},
    )
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Stubbed model container (predictions / history tests)
# --------------------------------------------------------------------------- #


class FakeInferenceEngine:
    """A deterministic inference engine stub for API tests."""

    def __init__(self) -> None:
        self.calls = 0

    async def predict(self, lon, lat, *, year=None, season=None):
        self.calls += 1
        return {
            "recommended_crop": "Paddy",
            "crop_probs": {"Paddy": 0.8, "Wheat": 0.2},
            "expected_yield": 6.12,
            "confidence": 0.8,
            "model_version": "test-1.0",
            "inference_time_ms": 12.5,
            "fallback": False,
            "raw_sample": None,
            "observation": None,
        }

    def status(self) -> dict:
        return {
            "ready": True,
            "model_version": "test-1.0",
            "queue_size": 0,
            "cache_enabled": True,
            "device": "cpu",
        }

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def warmup(self) -> None:
        pass


class FakeExplainabilityService:
    def summarize(self, *, sample=None, observation=None):
        return {"message": "fake explanation summary", "top_features": [["rainfall", 0.4]]}


@pytest.fixture
def app_with_fake_engine(app):
    """The app with a stubbed inference + explainability engine."""
    container = app.state.container
    container.model.override("inference_engine", FakeInferenceEngine())
    container.model.override("explainability_service", FakeExplainabilityService())
    return app


@pytest.fixture
def client_with_fake_engine(app_with_fake_engine):
    from starlette.testclient import TestClient

    with TestClient(app_with_fake_engine) as c:
        yield c
