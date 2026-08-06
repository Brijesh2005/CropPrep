"""Tests for the temporal / sequence feature builder."""

from __future__ import annotations

import pytest

from training.feature_engineering.temporal import TemporalFeatureBuilder


def test_temporal_features(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    row = TemporalFeatureBuilder().build(accepted[0])
    assert row["tmp.year"] == accepted[0].temporal.year
    assert row["tmp.season"] == accepted[0].temporal.season
    assert "tmp.date_count" in row
    assert "tmp.coverage_ratio" in row
    assert "tmp.mean_gap_days" in row


def test_observation_dates_joined_string(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    row = TemporalFeatureBuilder().build(accepted[0])
    value = row["tmp.observation_dates"]
    assert isinstance(value, str)
    assert value == "" or ";" in value or "-" in value


def test_without_dates_disabled(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    builder = TemporalFeatureBuilder(_cfg(include_dates=False))
    row = builder.build(accepted[0])
    assert "tmp.observation_dates" not in row


def test_empty_sequence_observation():
    from training.stam.observation import (
        AgriculturalObservation,
        LocationInfo,
        QualityReport,
        SequenceInfo,
        TabularFeatures,
        TemporalInfo,
    )

    obs = AgriculturalObservation(
        location=LocationInfo(lon=74.8, lat=13.1),
        temporal=TemporalInfo(year=2020),
        tabular=TabularFeatures(),
        sequence=SequenceInfo(),
        quality=QualityReport(passed=False, overall_score=0.0),
    )
    row = TemporalFeatureBuilder().build(obs)
    assert row["tmp.date_count"] == 0
    assert row["tmp.coverage_ratio"] is None
    assert row["tmp.mean_gap_days"] is None


def _cfg(**kwargs):
    from training.feature_engineering.config import TemporalFeatureConfig

    return TemporalFeatureConfig(**kwargs)
