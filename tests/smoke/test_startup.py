"""Smoke tests: application boots and core endpoints respond."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # backend/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_app(tmp_path):
    from app.core.config import load_settings
    from app.main import create_app

    settings = load_settings(
        env={
            "BACKEND_ENVIRONMENT": "test",
            "BACKEND_DATABASE__URL": f"sqlite+aiosqlite:///{tmp_path}/smoke.db",
            "BACKEND_MODEL__WARMUP": "false",
            "BACKEND_DATASET__VALIDATE_ON_STARTUP": "false",
            "BACKEND_RATE_LIMIT__ENABLED": "false",
            "BACKEND_LOG__JSON_LOGS": "false",
            "BACKEND_INFERENCE__ENABLE_FALLBACK": "true",
        }
    )
    return create_app(settings)


@pytest.fixture(scope="module")
def smoke_client(tmp_path_factory):
    from starlette.testclient import TestClient

    base = tmp_path_factory.mktemp("smoke")
    with TestClient(_make_app(base)) as client:
        yield client


def test_live_endpoint(smoke_client):
    r = smoke_client.get("/live")
    assert r.status_code == 200


def test_health_endpoint(smoke_client):
    r = smoke_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "healthy", "degraded")


def test_ready_endpoint(smoke_client):
    r = smoke_client.get("/ready")
    assert r.status_code == 200


def test_root_meta(smoke_client):
    r = smoke_client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "CropFusion Backend"


def test_openapi_generated(smoke_client):
    r = smoke_client.get("/openapi.json")
    assert r.status_code == 200
    assert "/api/v1/predict" in r.json()["paths"]


def test_security_headers_present(smoke_client):
    r = smoke_client.get("/")
    headers = {k.lower(): v for k, v in r.headers.items()}
    assert "x-content-type-options" in headers
    assert "x-frame-options" in headers


def test_request_id_returned(smoke_client):
    r = smoke_client.get("/live", headers={"X-Request-ID": "smoke-abc-123"})
    assert r.headers.get("x-request-id") == "smoke-abc-123"


def test_correlation_log_headers(smoke_client):
    r = smoke_client.get("/live")
    assert r.headers.get("x-request-id"), "missing request id header"
