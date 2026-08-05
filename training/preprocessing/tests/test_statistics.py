"""Tests for the statistics module."""

from __future__ import annotations

from training.preprocessing.statistics import DatasetStatistics, StatisticsReport


class _Tabular:
    def __init__(self, fields):
        self.fields = fields


class _Obs:
    def __init__(self, obs_id, crop, yield_value, year, fields, n_obs, score=80.0):
        self.observation_id = obs_id
        self.crop = crop
        self.yield_value = yield_value
        self.tabular = _Tabular(fields)
        self._n = n_obs

    def num_observations(self):
        return self._n


def _obs_list():
    return [
        _Obs("a", "Rice", 5200, 2020, {"rainfall_mm": 2100, "village": "A"}, 3),
        _Obs("b", "Rice", 5400, 2020, {"rainfall_mm": None, "village": "A"}, 2),
        _Obs("c", "Coconut", 3100, 2021, {"rainfall_mm": 2300}, 1),
    ]


def test_summarize_fields():
    report = DatasetStatistics.summarize(_obs_list())
    assert report.total_observations == 3
    assert report.class_distribution == {"Rice": 2, "Coconut": 1}
    assert report.yield_distribution["min"] == 3100.0
    assert report.yield_distribution["max"] == 5400.0
    assert report.sequence_length_distribution["min"] == 1.0
    assert report.sequence_length_distribution["max"] == 3.0
    assert report.missing_values["rainfall_mm"] == 1
    assert report.feature_statistics["rainfall_mm"]["mean"] == pytest_approx(2200.0)


def test_summarize_empty():
    report = DatasetStatistics.summarize([])
    assert report.total_observations == 0
    assert report.class_distribution == {}


def test_report_to_dict_and_save(tmp_path):
    report = DatasetStatistics.summarize(_obs_list())
    data = report.to_dict()
    assert data["total_observations"] == 3
    path = report.save(tmp_path)
    assert path.exists()
    import json

    assert json.loads(path.read_text(encoding="utf-8"))["total_observations"] == 3


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


def test_patch_statistics_with_extractor(observations, extractor, tmp_path):
    # Uses the real extractor to sample patch valid-ratios.
    report = DatasetStatistics.summarize(
        observations[:2], extractor=extractor, patch_size=32
    )
    assert report.patch_statistics["sampled"] > 0
