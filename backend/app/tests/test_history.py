"""History module tests."""

from __future__ import annotations


def test_history_after_predictions(client_with_fake_engine, auth_headers):
    for _ in range(2):
        r = client_with_fake_engine.post(
            "/api/v1/predict",
            headers=auth_headers,
            json={"lon": 74.8, "lat": 13.1},
        )
        assert r.status_code == 200

    r = client_with_fake_engine.get("/api/v1/predictions/history", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["items"][0]["recommended_crop"] == "Paddy"


def test_history_pagination(client_with_fake_engine, auth_headers):
    for _ in range(3):
        client_with_fake_engine.post(
            "/api/v1/predict", headers=auth_headers, json={"lon": 74.8, "lat": 13.1}
        )
    r = client_with_fake_engine.get(
        "/api/v1/predictions/history", headers=auth_headers, params={"limit": 2}
    )
    assert len(r.json()["items"]) == 2
