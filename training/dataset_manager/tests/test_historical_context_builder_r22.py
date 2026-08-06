"""Tests for :class:`HistoricalContextBuilderImpl` — the R2.2 multi-year,
per-location observation context (raw context only — no STAM inference).

Uses the fully-wired R2.2 manager fixture so spatial resolution, tabular
matching, satellite metadata and temporal persistence are all real.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from training.dataset_manager.exceptions import DatasetNotFoundError


def test_build_by_village_resolves_coordinates(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        village="Moodabidri", index_type="NDVI", resolution="R10m"
    )
    assert context.location == "Moodabidri"
    assert context.latitude == 13.08
    assert context.longitude == 74.89
    assert context.years == [2019, 2020]
    observation = next(o for o in context.observations if o.year == 2019)
    assert observation.tabular is not None
    assert observation.tabular["village"] == "Moodabidri"
    assert observation.tabular_source == "crop_yield"
    assert len(observation.ndvi) == 1
    assert observation.ndvi[0]["index_type"] == "NDVI"
    assert observation.quality["ndvi"] == 1


def test_build_by_district(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        district="Dakshina Kannada", index_type="NDVI"
    )
    assert context.location == "Dakshina Kannada"
    assert len(context.observations) >= 1


def test_build_by_raw_coordinates(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        latitude=13.08, longitude=74.89, index_type="NDVI"
    )
    assert context.years == [2019, 2020]
    assert context.latitude == 13.08


def test_build_multi_year(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        village="Moodabidri", index_type="NDVI"
    )
    years = {observation.year for observation in context.observations}
    assert years == {2019, 2020}
    assert context.missing_years == []


def test_build_returns_ndvi_and_evi_when_requested(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        village="Moodabidri", index_type=None
    )
    observation = context.observations[0]
    assert len(observation.ndvi) == 1
    assert len(observation.evi) == 1
    assert context.quality["total_records"] >= 2


def test_build_persists_temporal_records(r22_manager_factory):
    manager = r22_manager_factory()
    manager.build_historical_context(
        village="Moodabidri", index_type="NDVI"
    )
    temporal = manager.temporal_metadata(index_type="NDVI", year=2019)
    assert len(temporal) >= 1
    record = temporal[0]
    assert record["index_type"] == "NDVI"
    assert record["year"] == 2019
    assert record["count"] >= 1


def test_build_unknown_location_raises(r22_manager_factory):
    manager = r22_manager_factory()
    with pytest.raises(DatasetNotFoundError):
        manager.build_historical_context(
            village="Nowhereville", index_type="NDVI"
        )


def test_build_no_coordinates_raises(r22_manager_factory):
    manager = r22_manager_factory()
    with pytest.raises((DatasetNotFoundError, ValueError)):
        manager.build_historical_context(index_type="NDVI")


def test_quality_tracks_tabular_presence(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        village="Bantwal", index_type="NDVI"
    )
    assert context.quality["has_tabular"] is True
    assert context.quality["index_type"] == "NDVI"


def test_build_year_restriction(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        village="Moodabidri", index_type="NDVI", years=[2019]
    )
    assert context.years == [2019]
