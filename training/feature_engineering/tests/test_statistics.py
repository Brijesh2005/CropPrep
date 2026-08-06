"""Tests for corpus statistics."""

from __future__ import annotations

import json

from training.feature_engineering.statistics import CorpusStatistics


def test_summary_counts(corpus):
    stats = CorpusStatistics.summarize(corpus)
    assert stats.total == corpus.total
    assert stats.accepted_count + stats.rejected_count + stats.error_count == corpus.total
    assert stats.status_counts == corpus.status_counts()


def test_quality_and_crop_stats(corpus):
    stats = CorpusStatistics.summarize(corpus)
    assert stats.by_crop
    assert all(v > 0 for v in stats.by_crop.values())
    assert stats.quality["count"] == stats.accepted_count
    assert stats.quality["min"] >= 0.0
    assert stats.quality["max"] <= 100.0


def test_year_season_district(corpus):
    stats = CorpusStatistics.summarize(corpus)
    assert stats.by_year
    assert "2020" in stats.by_year or "2021" in stats.by_year
    assert stats.by_season
    assert "by_district" in stats.to_dict()
    assert stats.by_district  # village polygons resolve to district DK


def test_missing_labels(corpus):
    stats = CorpusStatistics.summarize(corpus)
    missing = stats.missing_labels
    assert set(missing) == {"crop", "yield"}
    assert missing["crop"] >= 0
    assert missing["yield"] >= 0


def test_save(tmp_path, corpus):
    path = CorpusStatistics.summarize(corpus).save(tmp_path)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["accepted"] == CorpusStatistics.summarize(corpus).accepted_count


def test_to_frame(corpus):
    frame = CorpusStatistics.summarize(corpus).to_frame()
    assert list(frame.columns) == ["crop", "count"]
    assert len(frame) == len(CorpusStatistics.summarize(corpus).by_crop)
