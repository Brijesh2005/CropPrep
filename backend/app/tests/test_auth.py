"""Authentication API tests."""

from __future__ import annotations


def test_register_login_me(client):
    r = client.post(
        "/api/v1/auth/register",
        json={"email": "farmer@crop.io", "password": "secret123", "full_name": "Farmer"},
    )
    assert r.status_code == 200
    assert r.json()["email"] == "farmer@crop.io"

    r = client.post(
        "/api/v1/auth/login",
        data={"username": "farmer@crop.io", "password": "secret123"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body and "refresh_token" in body

    r = client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert r.status_code == 200
    assert r.json()["role"] == "user"


def test_duplicate_register_returns_conflict(client):
    payload = {"email": "dup@crop.io", "password": "secret123"}
    assert client.post("/api/v1/auth/register", json=payload).status_code == 200
    r = client.post("/api/v1/auth/register", json=payload)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "B-AUTH-100"


def test_bad_login_returns_401(client):
    r = client.post(
        "/api/v1/auth/login", data={"username": "nobody@crop.io", "password": "wrong"}
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "B-AUTH-001"


def test_refresh_token(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "refresh@crop.io", "password": "secret123"},
    )
    login = client.post(
        "/api/v1/auth/login",
        data={"username": "refresh@crop.io", "password": "secret123"},
    ).json()
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_me_requires_token(client):
    r = client.get("/api/v1/users/me")
    assert r.status_code == 401
