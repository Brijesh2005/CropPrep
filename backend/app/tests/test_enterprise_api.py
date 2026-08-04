"""Phase 10 enterprise API tests.

These exercise the seeded enterprise app end-to-end: auth lifecycle (password
change/reset, email verification, sessions), user preferences/locations,
prediction-history search, notifications, feedback, analytics/audit, registry,
catalog, spatial, experiments and the config store.

The module is seeded once (roles, demo users, catalog, a minimal boundary set);
tests that mutate state use unique identifiers so they remain order-independent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.security import create_access_token

ENT_DB = "enterprise.db"


@pytest.fixture(scope="module")
def ent_settings(tmp_path_factory) -> "object":
    from app.core.config import load_settings

    tmp = tmp_path_factory.mktemp("ent")
    return load_settings(
        env={
            "BACKEND_DATABASE__URL": f"sqlite+aiosqlite:///{tmp / ENT_DB}",
            "BACKEND_ENVIRONMENT": "test",
            "BACKEND_RATE_LIMIT__ENABLED": "false",
            "BACKEND_MODEL__WARMUP": "false",
            "BACKEND_DATASET__VALIDATE_ON_STARTUP": "false",
            "BACKEND_SEED__ON_STARTUP": "true",
            "BACKEND_SEED__INCLUDE_BOUNDARIES": "true",
            "BACKEND_SEED__CSV_PATH": str(tmp / "does-not-exist.csv"),
        }
    )


@pytest.fixture(scope="module")
def ent_app(ent_settings):
    from app.main import create_app

    return create_app(ent_settings)


@pytest.fixture(scope="module")
def ent_client(ent_app):
    from starlette.testclient import TestClient

    with TestClient(ent_app) as c:
        yield c


@pytest.fixture(scope="module")
def ent_settings_obj(ent_app):
    return ent_app.state.container.config.resolve("settings")


@pytest.fixture(scope="module")
def db_url(ent_settings) -> str:
    return str(ent_settings.database.url)


# --------------------------------------------------------------------------- #
# Helpers (run async DB work against the file-backed SQLite via a fresh engine)
# --------------------------------------------------------------------------- #


def _run(db_url: str, fn) -> Any:
    """Run an async ``fn(session)`` against the file DB via a fresh engine."""

    async def _go():
        engine = create_async_engine(db_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with factory() as session:
                return await fn(session)
        finally:
            await engine.dispose()

    return asyncio.run(_go())


def _user_id(db_url: str, email: str) -> int:
    from sqlalchemy import text

    async def _lookup(session):
        result = await session.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email})
        row = result.fetchone()
        return row[0] if row else None

    return _run(db_url, _lookup)


def _token(ent_settings_obj, user_id: int, role: str) -> dict[str, str]:
    token = create_access_token(
        str(user_id), role=role, settings=ent_settings_obj.security
    )
    return {"Authorization": f"Bearer {token}"}


def _insert_prediction(db_url: str, *, user_id: int, **kwargs) -> int:
    from app.models.prediction import Prediction

    defaults = dict(
        location_lon=74.8, location_lat=13.0, location_name="Test Village",
        crop="Paddy", crop_probs={"Paddy": 0.9}, yield_prediction=6.5,
        confidence=0.85, model_version="test-1.0", source="point",
    )
    defaults.update(kwargs)

    async def _add(session):
        prediction = Prediction(user_id=user_id, **defaults)
        session.add(prediction)
        await session.commit()
        await session.refresh(prediction)
        return prediction.id

    return _run(db_url, _add)


def _enterprise_login(db_url: str, email: str, password: str, settings_obj) -> dict:
    """Login through the Phase 10 AuthService (creates a persisted session)."""

    async def _do(session):
        from database.services.auth import AuthService
        from database.services.redis_store import MemoryStore

        service = AuthService(
            session,
            settings_obj.security,
            settings_obj.security.password_policy,
            MemoryStore(settings_obj.redis),
        )
        return await service.login(email=email, password=password)

    return _run(db_url, _do)


# --------------------------------------------------------------------------- #
# Token fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def admin_headers(ent_settings_obj, db_url) -> dict[str, str]:
    user_id = _user_id(db_url, "admin@cropfusion.local")
    assert user_id, "seeded super admin not found"
    return _token(ent_settings_obj, user_id, "super_admin")


@pytest.fixture(scope="module")
def analyst_headers(ent_settings_obj, db_url) -> dict[str, str]:
    user_id = _user_id(db_url, "researcher@cropfusion.local")
    return _token(ent_settings_obj, user_id, "analyst")


@pytest.fixture(scope="module")
def farmer_headers(ent_settings_obj, db_url) -> dict[str, str]:
    user_id = _user_id(db_url, "farmer@cropfusion.local")
    return _token(ent_settings_obj, user_id, "user")


# --------------------------------------------------------------------------- #
# Auth: password lifecycle
# --------------------------------------------------------------------------- #


def test_password_change_lifecycle(ent_client, db_url):
    # A fresh Phase 8 user (pbkdf2 hash) can use the enterprise password change.
    ent_client.post(
        "/api/v1/auth/register",
        json={"email": "pwuser@crop.io", "password": "secret123", "full_name": "PW User"},
    )
    login = ent_client.post(
        "/api/v1/auth/login",
        data={"username": "pwuser@crop.io", "password": "secret123"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    wrong = ent_client.post(
        "/api/v1/auth/password/change",
        json={"current_password": "nope", "new_password": "NewPass123!"},
        headers=headers,
    )
    assert wrong.status_code == 401

    ok = ent_client.post(
        "/api/v1/auth/password/change",
        json={"current_password": "secret123", "new_password": "NewPass123!"},
        headers=headers,
    )
    assert ok.status_code == 200
    assert "revoked" in ok.json()["message"]

    # Old refresh tokens are revoked; a new login works with the new password.
    relogin = ent_client.post(
        "/api/v1/auth/login",
        data={"username": "pwuser@crop.io", "password": "NewPass123!"},
    )
    assert relogin.status_code == 200


def test_password_reset_flow(ent_client, admin_headers, db_url):
    email = "pwuser@crop.io"
    user_id = _user_id(db_url, email)
    assert user_id
    admin = ent_client.post(
        "/api/v1/notifications",
        json={"user_id": user_id, "notification_type": "system", "subject": "hello"},
        headers=admin_headers,
    )
    assert admin.status_code == 200

    req = ent_client.post("/api/v1/auth/password/reset", json={"email": email})
    assert req.status_code == 200
    assert req.json()["ok"] is True
    token = req.json()["token"]

    confirm = ent_client.post(
        "/api/v1/auth/password/reset/confirm",
        json={"token": token, "new_password": "ResetPass123!"},
    )
    assert confirm.status_code == 200

    login = ent_client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": "ResetPass123!"},
    )
    assert login.status_code == 200

    # Reset tokens are single-use.
    again = ent_client.post(
        "/api/v1/auth/password/reset/confirm",
        json={"token": token, "new_password": "Another123!"},
    )
    assert again.status_code == 401


def test_email_verification_flow(ent_client, db_url):
    ent_client.post(
        "/api/v1/auth/register",
        json={"email": "verify@crop.io", "password": "secret123", "full_name": "V User"},
    )
    login = ent_client.post(
        "/api/v1/auth/login", data={"username": "verify@crop.io", "password": "secret123"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    request_token = ent_client.post("/api/v1/auth/verify-email/request", headers=headers)
    assert request_token.status_code == 200
    token = request_token.json()["token"]

    confirm = ent_client.post(
        "/api/v1/auth/verify-email/confirm", json={"token": token}
    )
    assert confirm.status_code == 200
    assert confirm.json()["email"] == "verify@crop.io"


def test_sessions_list_and_revoke(ent_client, ent_settings_obj, db_url):
    ent_client.post(
        "/api/v1/auth/register",
        json={"email": "sess@crop.io", "password": "secret123", "full_name": "S User"},
    )
    # Phase 10 login persists real session rows (the Phase 8 /auth/login does not).
    first = _enterprise_login(db_url, "sess@crop.io", "secret123", ent_settings_obj)
    second = _enterprise_login(db_url, "sess@crop.io", "secret123", ent_settings_obj)
    headers = {"Authorization": f"Bearer {second['access_token']}"}

    sessions = ent_client.get("/api/v1/auth/sessions", headers=headers)
    assert sessions.status_code == 200
    items = sessions.json()
    assert len(items) >= 2
    target = items[0]["session_id"]

    revoked = ent_client.delete(f"/api/v1/auth/sessions/{target}", headers=headers)
    assert revoked.status_code == 200
    assert revoked.json()["revoked"] is True

    remaining = ent_client.get("/api/v1/auth/sessions", headers=headers)
    assert all(s["session_id"] != target for s in remaining.json())


def test_rbac_denies_regular_users(ent_client, farmer_headers):
    r = ent_client.get("/api/v1/admin/enterprise/dashboard", headers=farmer_headers)
    assert r.status_code == 403
    r = ent_client.post(
        "/api/v1/registry/models",
        json={"name": "nope", "version": "0.1"},
        headers=farmer_headers,
    )
    assert r.status_code == 403
    r = ent_client.post("/api/v1/notifications", json={}, headers=farmer_headers)
    assert r.status_code == 403


# --------------------------------------------------------------------------- #
# Users: preferences + saved locations
# --------------------------------------------------------------------------- #


def test_preferences_roundtrip(ent_client, farmer_headers):
    put = ent_client.put(
        "/api/v1/users/preferences",
        json={"language": "kn", "theme": "dark", "units": "metric"},
        headers=farmer_headers,
    )
    assert put.status_code == 200
    data = put.json()
    assert data["preferred_language"] == "kn"
    assert data["theme"] == "dark"

    get = ent_client.get("/api/v1/users/preferences", headers=farmer_headers)
    assert get.status_code == 200
    assert get.json()["preferred_language"] == "kn"


def test_saved_locations(ent_client, farmer_headers):
    created = ent_client.post(
        "/api/v1/users/locations",
        json={"name": "Home Farm", "lon": 74.8, "lat": 13.0, "is_default": True},
        headers=farmer_headers,
    )
    assert created.status_code == 200
    loc_id = created.json()["id"]

    listed = ent_client.get("/api/v1/users/locations", headers=farmer_headers)
    assert listed.status_code == 200
    assert any(l["id"] == loc_id for l in listed.json())

    second = ent_client.post(
        "/api/v1/users/locations",
        json={"name": "Second Farm", "lon": 75.0, "lat": 12.5},
        headers=farmer_headers,
    )
    second_id = second.json()["id"]
    primary = ent_client.put(
        f"/api/v1/users/locations/{second_id}/primary", headers=farmer_headers
    )
    assert primary.status_code == 200
    assert primary.json()["ok"] is True

    deleted = ent_client.delete(
        f"/api/v1/users/locations/{loc_id}", headers=farmer_headers
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True


# --------------------------------------------------------------------------- #
# Prediction history search
# --------------------------------------------------------------------------- #


def test_prediction_history_search(ent_client, farmer_headers, db_url):
    user_id = _user_id(db_url, "farmer@cropfusion.local")
    _insert_prediction(
        db_url, user_id=user_id, crop="Paddy", season="Kharif", year=2024,
        district="Mangalore", confidence=0.9,
    )
    _insert_prediction(
        db_url, user_id=user_id, crop="Wheat", season="Rabi", year=2023,
        district="Pune", confidence=0.5,
    )

    all_rows = ent_client.get(
        "/api/v1/predictions/history/search", headers=farmer_headers
    )
    assert all_rows.status_code == 200
    assert all_rows.json()["total"] >= 2

    filtered = ent_client.get(
        "/api/v1/predictions/history/search?crop=Paddy", headers=farmer_headers
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] >= 1
    assert all(i["crop"] == "Paddy" for i in filtered.json()["items"])

    confident = ent_client.get(
        "/api/v1/predictions/history/search?min_confidence=0.75", headers=farmer_headers
    )
    assert confident.status_code == 200
    assert all(i["confidence"] >= 0.75 for i in confident.json()["items"])


# --------------------------------------------------------------------------- #
# Notifications
# --------------------------------------------------------------------------- #


def test_notifications_lifecycle(ent_client, admin_headers, farmer_headers, db_url):
    farmer_id = _user_id(db_url, "farmer@cropfusion.local")
    for n in range(2):
        r = ent_client.post(
            "/api/v1/notifications",
            json={
                "user_id": farmer_id, "notification_type": "seasonal",
                "subject": f"notice {n}", "body": "rainfall alert",
            },
            headers=admin_headers,
        )
        assert r.status_code == 200

    listed = ent_client.get("/api/v1/notifications", headers=farmer_headers)
    assert listed.status_code == 200
    assert listed.json()["total"] >= 2

    unread = ent_client.get("/api/v1/notifications/unread-count", headers=farmer_headers)
    assert unread.status_code == 200
    assert unread.json()["unread"] >= 2

    first_id = listed.json()["items"][0]["id"]
    read = ent_client.post(f"/api/v1/notifications/{first_id}/read", headers=farmer_headers)
    assert read.status_code == 200
    assert read.json()["ok"] is True

    all_read = ent_client.post("/api/v1/notifications/read-all", headers=farmer_headers)
    assert all_read.status_code == 200
    assert all_read.json()["marked"] >= 1
    unread_after = ent_client.get(
        "/api/v1/notifications/unread-count", headers=farmer_headers
    )
    assert unread_after.json()["unread"] == 0


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #


def test_feedback_flow(ent_client, admin_headers, farmer_headers):
    submit = ent_client.post(
        "/api/v1/feedback",
        json={"rating": 4, "category": "general", "comment": "works well"},
        headers=farmer_headers,
    )
    assert submit.status_code == 200
    feedback_id = submit.json()["id"]

    listed = ent_client.get("/api/v1/feedback", headers=admin_headers)
    assert listed.status_code == 200
    assert any(f["id"] == feedback_id for f in listed.json()["items"])

    resolve = ent_client.post(
        f"/api/v1/feedback/{feedback_id}/resolve",
        json={"note": "acknowledged"},
        headers=admin_headers,
    )
    assert resolve.status_code == 200
    assert resolve.json()["ok"] is True


def test_feedback_validates_rating(ent_client, farmer_headers):
    bad = ent_client.post(
        "/api/v1/feedback", json={"rating": 9, "category": "general"}, headers=farmer_headers
    )
    assert bad.status_code == 422


# --------------------------------------------------------------------------- #
# Analytics + audit (admin)
# --------------------------------------------------------------------------- #


def test_enterprise_dashboard(ent_client, admin_headers):
    r = ent_client.get("/api/v1/admin/enterprise/dashboard", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["users"] >= 4
    assert "predictions" in body["totals"]
    assert "by_crop" in body


def test_audit_trail(ent_client, admin_headers):
    r = ent_client.get("/api/v1/admin/enterprise/audit", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert len(body["items"]) >= 1
    assert all(i["action"] and i["created_at"] for i in body["items"])


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_model_registry(ent_client, admin_headers):
    created = ent_client.post(
        "/api/v1/registry/models",
        json={"name": "yieldnet", "version": "1.2.0", "is_active": True},
        headers=admin_headers,
    )
    assert created.status_code == 200
    assert created.json()["version"] == "1.2.0"

    listed = ent_client.get("/api/v1/registry/models", headers=admin_headers)
    assert listed.status_code == 200
    assert any(m["name"] == "yieldnet" for m in listed.json()["items"])

    activated = ent_client.post(
        "/api/v1/registry/models/activate?name=yieldnet&version=1.2.0",
        headers=admin_headers,
    )
    assert activated.status_code == 200
    assert activated.json()["ok"] is True

    dup = ent_client.post(
        "/api/v1/registry/models",
        json={"name": "yieldnet", "version": "1.2.0"},
        headers=admin_headers,
    )
    assert dup.status_code == 400


def test_dataset_registry(ent_client, admin_headers):
    created = ent_client.post(
        "/api/v1/registry/datasets",
        json={"name": "crop-yield", "version": "2024.1"},
        headers=admin_headers,
    )
    assert created.status_code == 200

    validated = ent_client.post(
        "/api/v1/registry/datasets/validate?name=crop-yield&version=2024.1&status=valid",
        headers=admin_headers,
    )
    assert validated.status_code == 200
    assert validated.json()["validation_status"] == "valid"


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def test_catalog_crops(ent_client, admin_headers):
    listed = ent_client.get("/api/v1/catalog/crops")
    assert listed.status_code == 200
    assert len(listed.json()["items"]) >= 1

    created = ent_client.post(
        "/api/v1/catalog/crops",
        json={"code": "BG-2024", "name": "Black Gram", "category": "pulse"},
        headers=admin_headers,
    )
    assert created.status_code == 200

    searched = ent_client.get("/api/v1/catalog/crops?search=Black")
    assert searched.status_code == 200
    assert any(c["code"] == "BG-2024" for c in searched.json()["items"])

    dup = ent_client.post(
        "/api/v1/catalog/crops",
        json={"code": "BG-2024", "name": "Duplicate"},
        headers=admin_headers,
    )
    assert dup.status_code == 400


def test_catalog_seasons(ent_client):
    listed = ent_client.get("/api/v1/catalog/seasons")
    assert listed.status_code == 200
    assert len(listed.json()["items"]) >= 1


# --------------------------------------------------------------------------- #
# Spatial
# --------------------------------------------------------------------------- #


def test_spatial_resolve_and_boundaries(ent_client):
    # The single synthetic district is centred around (68.5, 8.0).
    resolved = ent_client.get("/api/v1/spatial/resolve?lon=68.5&lat=8.0")
    assert resolved.status_code == 200
    body = resolved.json()
    assert body is not None
    assert body["district"] is not None

    boundaries = ent_client.get("/api/v1/spatial/boundaries?level=district")
    assert boundaries.status_code == 200
    assert len(boundaries.json()) >= 1

    counts = ent_client.get("/api/v1/spatial/boundaries/counts")
    assert counts.status_code == 200
    assert counts.json()["district"] >= 1


def test_spatial_locations(ent_client, admin_headers):
    created = ent_client.post(
        "/api/v1/spatial/locations",
        json={"name": "Station 1", "lon": 68.5, "lat": 8.0, "location_type": "station"},
        headers=admin_headers,
    )
    assert created.status_code == 200
    assert created.json()["id"]

    nearest = ent_client.get("/api/v1/spatial/locations?lon=68.5&lat=8.0&radius_km=50")
    assert nearest.status_code == 200
    assert any(l["name"] == "Station 1" for l in nearest.json())


def test_spatial_resolve_admin_payload(ent_client, admin_headers):
    r = ent_client.post(
        "/api/v1/spatial/resolve/admin",
        json={"lon": 68.5, "lat": 8.0},
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert r.json()["district"] is not None


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #


def test_experiments_lifecycle(ent_client, admin_headers):
    created = ent_client.post(
        "/api/v1/experiments",
        json={"name": "exp-kharif", "config": {"epochs": 10}},
        headers=admin_headers,
    )
    assert created.status_code == 200
    exp_id = created.json()["id"]

    started = ent_client.post(f"/api/v1/experiments/{exp_id}/start", headers=admin_headers)
    assert started.status_code == 200
    assert started.json()["status"] == "running"

    finished = ent_client.post(
        f"/api/v1/experiments/{exp_id}/finish",
        json={"metrics": {"mae": 0.31}},
        headers=admin_headers,
    )
    assert finished.status_code == 200
    assert finished.json()["status"] == "completed"

    listed = ent_client.get("/api/v1/experiments", headers=admin_headers)
    assert listed.status_code == 200
    assert any(e["id"] == exp_id for e in listed.json()["items"])


# --------------------------------------------------------------------------- #
# Config store
# --------------------------------------------------------------------------- #


def test_config_store(ent_client, admin_headers):
    put = ent_client.put(
        "/api/v1/config-store",
        json={"key": "crop.alert.threshold", "value": {"rainfall": 50}, "category": "crop"},
        headers=admin_headers,
    )
    assert put.status_code == 200
    assert put.json()["version"] >= 1

    get = ent_client.get("/api/v1/config-store/crop.alert.threshold")
    assert get.status_code == 200
    assert get.json()["value"] == {"rainfall": 50}

    bump = ent_client.put(
        "/api/v1/config-store",
        json={"key": "crop.alert.threshold", "value": {"rainfall": 60}, "category": "crop"},
        headers=admin_headers,
    )
    assert bump.json()["version"] == 2
