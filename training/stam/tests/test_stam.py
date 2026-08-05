"""End-to-end tests for the STAM facade over the synthetic dataset."""

from __future__ import annotations

import pytest

from training.stam.exceptions import (
    LocationNotFoundError,
    NotInitializedError,
)
from training.stam.observation import AgriculturalObservation
from training.stam.stam import STAM


def test_not_initialized_raises(manager, stam_config):
    stam = STAM(manager, stam_config)
    with pytest.raises(NotInitializedError):
        stam.build_observation(74.802, 13.098, year=2020, season="Kharif")


def test_initialize_and_summary(stam):
    summary = stam.summary()
    assert summary["initialized"] is True
    assert summary["patch_size"] == 16
    assert summary["indexes"]["locations"] > 0
    assert summary["seasons"] == ["Kharif", "Rabi", "Summer"]


def test_build_observation_full(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    assert isinstance(obs, AgriculturalObservation)
    assert obs.location.admin.village == "A"
    assert obs.temporal.year == 2020
    assert obs.temporal.season == "Kharif"
    assert obs.crop == "Rice"
    assert obs.yield_value == 5200.0
    assert obs.num_observations() == 3
    assert obs.has_paired_images is True
    assert obs.quality.passed is True
    assert obs.sequence.resolution == "R10m"
    assert len(obs.sequence.ndvi_paths) == 3
    assert len(obs.sequence.evi_paths) == 3


def test_build_observation_serializable(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    payload = obs.model_dump(mode="json")
    assert payload["observation_id"]
    assert payload["temporal"]["year"] == 2020
    # Round-trips through the model.
    rebuilt = AgriculturalObservation.model_validate(payload)
    assert rebuilt.crop == "Rice"


def test_build_observation_train_dict(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    train = obs.to_train_dict()
    assert train["crop"] == "Rice"
    assert train["yield_value"] == 5200.0
    assert len(train["ndvi_paths"]) == 3
    assert len(train["evi_paths"]) == 3
    assert train["quality_score"] == obs.quality.overall_score


def test_build_observation_year_default(stam):
    # Without year/season: year defaults to the latest tabular year (2021),
    # whose sequence has NDVI only -> strict pairing falls back and flags it.
    obs = stam.build_observation(74.802, 13.098)
    assert obs.temporal.year == 2021
    assert obs.quality.passed is False


def test_build_observation_farmer_path_resolves_season(stam):
    # Farmer path: location only — the season is inferred from the date and a
    # multi-year historical context (same location + same season) is attached.
    obs = stam.build_observation(74.802, 13.098)
    assert obs.temporal.season is not None
    assert obs.temporal.season in ["Kharif", "Rabi", "Summer"]
    assert obs.historical_context is not None
    assert obs.historical_context.season == obs.temporal.season
    assert obs.historical_context.resolved_year == obs.temporal.year
    assert obs.historical_context.years == [2019, 2020, 2021]
    assert obs.historical_context.total_records > 0
    assert obs.historical_context.source == "dataset_manager"
    assert obs.dataset_version == obs.historical_context.dataset_version
    assert obs.season_calendar_version == obs.historical_context.season_calendar_version
    assert obs.provenance["historical_context_years"] == [2019, 2020, 2021]


def test_build_observation_research_path_keeps_versions(stam):
    # Research path: explicit year/season still resolves the season and
    # records the calendar + dataset versions on the observation.
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    assert obs.temporal.year == 2020
    assert obs.temporal.season == "Kharif"
    assert obs.season_calendar_version is not None
    assert obs.dataset_version is None or isinstance(obs.dataset_version, str)
    assert obs.historical_context.years == [2019, 2020, 2021]


def test_build_observation_missing_evi_flags(stam):
    obs = stam.build_observation(74.802, 13.098, year=2021, season="Kharif")
    assert obs.num_observations() == 1
    assert any(i.code == "ST-Q-PAIR-002" for i in obs.quality.issues)


def test_build_observation_location_not_found(stam):
    with pytest.raises(LocationNotFoundError):
        stam.build_observation(78.0, 20.0, year=2020, season="Kharif")


def test_find_nearest(stam):
    nearest = stam.find_nearest(74.802, 13.098)
    assert nearest["distance_km"] < 1.0
    assert nearest["id"]


def test_build_sequence(stam):
    result = stam.build_sequence(74.802, 13.098, year=2020, season="Kharif")
    assert result.observation_count == 3
    assert result.paired_count == 3
    assert result.sequence.sorted_dates[0].year == 2020


def test_get_patch(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    ndvi_path = obs.sequence.pairs[0].ndvi.path
    patch = stam.get_patch(ndvi_path, 74.802, 13.098, size=16)
    assert patch.shape == (16, 16)
    assert patch.valid_ratio > 0.5
    assert patch.crs == "EPSG:4326"


def test_validate_reports(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    report = stam.validate(obs)
    assert report.passed is True
    assert report.overall_score == obs.quality.overall_score


def test_observation_cache_hit(stam, manager):
    stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    key = stam.cache.observation_key(74.802, 13.098, 2020, "Kharif")
    cached = manager.cache_get(key)
    assert cached is not None
    assert cached["crop"] == "Rice"


def test_find_nearest_before_initialize_raises(manager, stam_config):
    stam = STAM(manager, stam_config)
    with pytest.raises(NotInitializedError):
        stam.find_nearest(74.802, 13.098)
