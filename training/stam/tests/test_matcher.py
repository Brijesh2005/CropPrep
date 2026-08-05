"""Integration tests for the spatial-temporal matcher over a real Dataset Manager."""

from __future__ import annotations

import pytest

from training.stam.exceptions import (
    InvalidCoordinatesError,
    LocationNotFoundError,
)
from training.stam.matcher import SpatialTemporalMatcher


@pytest.fixture
def matcher(manager, stam_config):
    return SpatialTemporalMatcher(manager, stam_config).initialize()


def test_find_nearest(matcher):
    result = matcher.find_nearest(74.802, 13.098)
    assert result.distance_km < 1.0
    assert result.point.name != ""  # a boundary or image location


def test_find_nearest_out_of_radius(matcher):
    with pytest.raises(LocationNotFoundError):
        matcher.find_nearest(78.0, 20.0)  # far away


def test_invalid_coordinates(matcher):
    with pytest.raises(InvalidCoordinatesError):
        matcher.find_nearest(200.0, 13.0)


def test_resolve_admin(matcher):
    admin = matcher.resolve_admin(74.802, 13.098)
    assert admin is not None
    assert admin.village == "A"
    assert admin.district == "DK"
    assert admin.taluk == "T1"
    assert admin.level == "village"


def test_location_info(matcher):
    info = matcher.location_info(74.802, 13.098)
    assert info.admin is not None and info.admin.village == "A"
    assert info.distance_km is not None
    assert info.dataset_location_id is not None


def test_resolve_temporal_kharif(matcher):
    context = matcher.resolve_temporal(year=2020, season="Kharif")
    assert context.year == 2020
    assert context.season is not None
    assert context.season.name == "Kharif"
    assert context.planting_start.year == 2020


def test_resolve_temporal_defaults_to_latest_year(matcher):
    context = matcher.resolve_temporal()  # no year/season
    # Latest tabular year in the synthetic table is 2021.
    assert context.year == 2021


def test_match_tabular_village(matcher):
    tabular = matcher.match_tabular(
        village="A", district="DK", year=2020, season="Kharif"
    )
    assert tabular is not None
    assert tabular.crop == "Rice"
    assert tabular.yield_value == 5200.0
    assert tabular.matched_level == "village"


def test_match_tabular_district_fallback(matcher):
    tabular = matcher.match_tabular(
        village="UnknownVillage", district="DK", year=2020, season="Kharif"
    )
    assert tabular is not None
    assert tabular.matched_level == "district"


def test_match_images_season_filtered(matcher):
    context = matcher.resolve_temporal(year=2020, season="Kharif")
    ndvi, evi = matcher.match_images(year=context.year, season=context.season)
    assert len(ndvi) == 3
    assert len(evi) == 3
    assert all(r.observation_date.year == 2020 for r in ndvi)


def test_match_images_respects_resolution(matcher, stam_config):
    context = matcher.resolve_temporal(year=2020, season="Kharif")
    ndvi, evi = matcher.match_images(
        year=context.year, season=context.season, resolution="R20m"
    )
    assert ndvi == [] and evi == []
