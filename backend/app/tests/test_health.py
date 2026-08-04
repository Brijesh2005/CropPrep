"""Health + configuration + monitoring API tests."""

from __future__ import annotations


def test_liveness(client):
    r = client.get("/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "checks" in r.json()


def test_readiness(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


def test_public_config(client):
    r = client.get("/api/v1/config")
    assert r.status_code == 200
    assert "app_name" in r.json()


def test_monitoring_requires_admin(client, auth_headers):
    r = client.get("/api/v1/monitoring/metrics", headers=auth_headers)
    assert r.status_code == 403  # a plain user is not an admin


def test_admin_endpoints_require_admin(client, auth_headers):
    r = client.get("/api/v1/admin/dashboard", headers=auth_headers)
    assert r.status_code == 403


def test_error_handling_structured(client):
    # Unknown path -> structured 404.
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    assert "error" in r.json()
    assert r.json()["error"]["code"] == "B-NOTFOUND-001"
