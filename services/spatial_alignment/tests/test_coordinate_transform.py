"""Unit tests for coordinate / CRS helpers."""

from __future__ import annotations

import pytest
from rasterio.transform import Affine, from_origin

from services.spatial_alignment.coordinate_transform import (
    crs_to_epsg,
    geographic_to_raster_index,
    normalise_crs,
    patch_window,
    pixel_to_world,
    transform_point,
    validate_crs,
    window_bounds,
    world_to_pixel,
)
from services.spatial_alignment.exceptions import CRSMismatchError
from services.spatial_alignment.coordinate_transform import assert_same_crs


def test_normalise_crs_forms():
    assert normalise_crs("EPSG:4326").to_epsg() == 4326
    assert normalise_crs(4326).to_epsg() == 4326
    assert normalise_crs("4326").to_epsg() == 4326
    assert normalise_crs(None) is None
    assert normalise_crs("not-a-crs") is None


def test_crs_to_epsg():
    assert crs_to_epsg("EPSG:32643") == 32643
    assert crs_to_epsg(None) is None


def test_validate_crs():
    assert validate_crs("EPSG:4326") is True
    assert validate_crs("garbage") is False


def test_assert_same_crs_mismatch_raises():
    import pytest

    with pytest.raises(CRSMismatchError):
        assert_same_crs("EPSG:4326", "EPSG:32643", context="test")
    with pytest.raises(CRSMismatchError):
        assert_same_crs("EPSG:4326", None, context="test")


def test_transform_point_roundtrip():
    x, y = transform_point("EPSG:4326", "EPSG:32643", 74.8, 13.1)
    lon, lat = transform_point("EPSG:32643", "EPSG:4326", x, y)
    assert abs(lon - 74.8) < 1e-6
    assert abs(lat - 13.1) < 1e-6


def test_world_pixel_roundtrip():
    transform = from_origin(74.80, 13.10, 0.0001, 0.0001)  # north-up
    row, col = world_to_pixel(transform, 74.8005, 13.0995)
    x, y = pixel_to_world(transform, row, col)
    assert abs(x - 74.8005) < 0.00011
    assert abs(y - 13.0995) < 0.00011


def test_geographic_to_raster_index():
    transform = from_origin(74.80, 13.10, 0.0001, 0.0001)
    # 5 pixels east and 5 pixels south of the top-left origin. Float
    # representation can floor the row to 4, so allow within-one tolerance.
    row, col = geographic_to_raster_index(transform, "EPSG:4326", 74.8005, 13.0995)
    assert col == 5
    assert abs(row - 5) <= 1


def test_patch_window_centred():
    transform = from_origin(74.80, 13.10, 0.0001, 0.0001)
    window = patch_window(transform, 10, 10, size=8)
    assert window.row_off == 6
    assert window.col_off == 6
    assert window.height == 8 and window.width == 8


def test_window_bounds():
    transform = from_origin(74.80, 13.10, 0.0001, 0.0001)
    # A 2x2 window centred on pixel (0,0) spans cols/rows -1..1, so its top
    # edge is one pixel above the raster origin (13.10 + 0.0001).
    window = patch_window(transform, 0, 0, size=2)
    left, bottom, right, top = window_bounds(transform, window)
    assert left == pytest.approx(74.7999, rel=1e-6)
    assert top == pytest.approx(13.1001, rel=1e-6)
    assert right == pytest.approx(74.8001, rel=1e-6)
