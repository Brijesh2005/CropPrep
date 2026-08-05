"""Predictions API tests (stubbed inference engine)."""

from __future__ import annotations


def test_predict_anonymous(client_with_fake_engine):
    r = client_with_fake_engine.post(
        "/api/v1/predict", json={"lon": 74.8, "lat": 13.1}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_crop"] == "Paddy"
    assert body["confidence"] == 0.8
    assert body["prediction_id"] is not None


def test_predict_with_explanation(client_with_fake_engine, auth_headers):
    r = client_with_fake_engine.post(
        "/api/v1/predict",
        headers=auth_headers,
        json={"lon": 74.8, "lat": 13.1, "include_explanation": True},
    )
    assert r.status_code == 200
    assert r.json()["explanation_summary"] is not None


def test_predict_map(client_with_fake_engine, auth_headers):
    r = client_with_fake_engine.post(
        "/api/v1/predict/map",
        headers=auth_headers,
        json={"points": [{"lon": 74.8, "lat": 13.1}, {"lon": 74.9, "lat": 13.2}]},
    )
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_predict_invalid_coordinates(client_with_fake_engine, auth_headers):
    r = client_with_fake_engine.post(
        "/api/v1/predict", headers=auth_headers, json={"lon": 999, "lat": 13.1}
    )
    assert r.status_code == 422


def test_predict_location_alias(client_with_fake_engine):
    r = client_with_fake_engine.post(
        "/api/v1/predict/location", json={"lon": 74.8, "lat": 13.1}
    )
    assert r.status_code == 200
    assert r.json()["recommended_crop"] == "Paddy"


def test_history_requires_auth(client_with_fake_engine):
    r = client_with_fake_engine.get("/api/v1/predictions/history")
    assert r.status_code == 401
