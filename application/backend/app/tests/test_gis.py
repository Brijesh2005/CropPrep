"""GIS service + API tests."""

from __future__ import annotations

import pytest

from app.core.exceptions import GISError, NotFoundError
from app.modules.gis.service import GISService, Location


def _locations():
    return [
        Location(id="v1", lon=74.801, lat=13.099, name="Village A", admin={"village": "Village A", "district": "D1"}),
        Location(id="v2", lon=74.900, lat=13.200, name="Village B", admin={"village": "Village B", "district": "D2"}),
    ]


def test_nearest_location():
    service = GISService(_locations())
    result = service.nearest(74.802, 13.100, k=1)
    assert result[0].id == "v1"
    assert result[0].distance_km < 1.0


def test_nearest_ranking():
    service = GISService(_locations())
    result = service.nearest(74.805, 13.100, k=2)
    assert result[0].id == "v1"
    assert result[1].id == "v2"
    assert result[0].distance_km <= result[1].distance_km


def test_invalid_coordinates_raise():
    service = GISService(_locations())
    with pytest.raises(GISError):
        service.nearest(200.0, 13.0)


def test_get_and_search():
    service = GISService(_locations())
    assert service.get("v1").name == "Village A"
    with pytest.raises(NotFoundError):
        service.get("nope")
    matches = service.search("village b")
    assert len(matches) == 1 and matches[0].id == "v2"


def test_only_dataset_locations_returned():
    # Locations outside the dataset are simply absent.
    service = GISService(_locations())
    result = service.nearest(180.0, -89.0, k=1)  # far from both
    assert result  # nearest of the available dataset locations, nothing else
    assert result[0].id in {"v1", "v2"}


def test_gis_api_locations(client):
    r = client.get("/api/v1/gis/locations")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
