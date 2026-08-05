"""Tests for quality filtering of observations."""

from __future__ import annotations

from training.preprocessing.config import QualityConfig
from training.preprocessing.validators import filter_observation, filter_observations


class _Location:
    def __init__(self, lon=74.8, lat=13.1):
        self.lon, self.lat = lon, lat


class _Quality:
    def __init__(self, score=90.0):
        self.overall_score = score


class _Obs:
    def __init__(self, obs_id="obs-1", lon=74.8, lat=13.1, score=90.0,
                 crop="Rice", yield_value=5200.0, observations=3, paired=True):
        self.observation_id = obs_id
        self.location = _Location(lon, lat)
        self.quality = _Quality(score)
        self.crop = crop
        self.yield_value = yield_value
        self._obs_count = observations
        self._paired = paired

    def num_observations(self):
        return self._obs_count

    @property
    def has_paired_images(self):
        return self._paired


def test_accepts_healthy():
    decision = filter_observation(_Obs(), QualityConfig())
    assert decision.accepted is True
    assert decision.reasons == []


def test_rejects_invalid_coordinates():
    decision = filter_observation(
        _Obs(lon=500.0), QualityConfig(require_valid_coordinates=True)
    )
    assert decision.accepted is False
    assert "invalid_coordinates" in decision.reasons


def test_rejects_low_quality():
    decision = filter_observation(
        _Obs(score=30.0), QualityConfig(min_quality_score=40.0)
    )
    assert decision.accepted is False
    assert any("quality_score" in r for r in decision.reasons)


def test_rejects_missing_labels():
    decision = filter_observation(
        _Obs(crop=None, yield_value=None),
        QualityConfig(require_crop_label=True, require_yield_label=True),
    )
    assert decision.accepted is False
    assert "missing_crop_label" in decision.reasons
    assert "missing_yield_label" in decision.reasons


def test_rejects_too_few_observations():
    decision = filter_observation(
        _Obs(observations=0), QualityConfig(min_observations=1)
    )
    assert decision.accepted is False
    assert any("too_few_observations" in r for r in decision.reasons)


def test_rejects_unpaired():
    decision = filter_observation(
        _Obs(paired=False), QualityConfig(reject_unpaired=True)
    )
    assert decision.accepted is False
    assert "unpaired_images" in decision.reasons


def test_filter_batch():
    obs = [_Obs("a", score=90), _Obs("b", score=10)]
    accepted, decisions = filter_observations(obs, QualityConfig(min_quality_score=40))
    assert len(accepted) == 1
    assert accepted[0].observation_id == "a"
    assert decisions[1].accepted is False
