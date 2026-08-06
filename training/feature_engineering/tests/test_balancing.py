"""Tests for the class-balancing report."""

from __future__ import annotations

import json

import pytest

from training.feature_engineering.balancing import BalancingReport


def test_counts_and_shares(corpus):
    report = BalancingReport.summarize(corpus)
    assert report.total == len(corpus.accepted())
    assert sum(report.class_counts.values()) == report.total
    assert sum(report.class_shares.values()) == pytest.approx(1.0, abs=1e-3)


def test_class_weights_normalised(corpus):
    report = BalancingReport.summarize(corpus)
    weights = report.class_weights()
    assert set(weights.keys()) == set(report.class_counts.keys())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(w > 0 for w in weights.values())


def test_imbalance_and_minority(corpus):
    report = BalancingReport.summarize(corpus)
    assert report.imbalance_ratio is None or report.imbalance_ratio >= 1.0
    if report.n_classes > 1:
        assert report.minority_classes or report.majority_classes
        assert report.recommended_strategy in {
            "balanced", "oversample_minority", "combined_weights_and_oversample",
        }


def test_sample_weights_ordered(corpus):
    report = BalancingReport.summarize(corpus)
    accepted = corpus.accepted_observations()
    if accepted:
        weights = report.sample_weights(accepted)
        assert len(weights) == len(accepted)
        assert all(w > 0 for w in weights)


def test_to_dict_save(tmp_path, corpus):
    report = BalancingReport.summarize(corpus)
    payload = report.to_dict()
    assert payload["total"] == report.total
    path = report.save(tmp_path)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["n_classes"] == report.n_classes
