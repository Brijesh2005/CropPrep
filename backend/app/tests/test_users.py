"""Users module tests."""

from __future__ import annotations


def test_update_profile(client, auth_headers):
    r = client.put(
        "/api/v1/users/profile",
        headers=auth_headers,
        json={"full_name": "Dr Farmer"},
    )
    assert r.status_code == 200
    assert r.json()["full_name"] == "Dr Farmer"


def test_update_profile_password(client, auth_headers):
    r = client.put(
        "/api/v1/users/profile", headers=auth_headers, json={"password": "newpassword1"}
    )
    assert r.status_code == 200
    # Old password no longer works; new one does.
    old = client.post(
        "/api/v1/auth/login", data={"username": "tester@crop.io", "password": "secret123"}
    )
    new = client.post(
        "/api/v1/auth/login", data={"username": "tester@crop.io", "password": "newpassword1"}
    )
    assert old.status_code == 401
    assert new.status_code == 200
