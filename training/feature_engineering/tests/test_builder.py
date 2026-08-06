"""Tests for the feature builder registry and feature-frame assembly."""

from __future__ import annotations

import pandas as pd
import pytest

from training.feature_engineering.builder import (
    FeatureBuilderRegistry,
    build_feature_frame,
    build_features,
)
from training.feature_engineering.config import FeatureEngineeringConfig
from training.feature_engineering.exceptions import FeatureFrameError


def test_registry_build_merges_modalities(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    row = FeatureBuilderRegistry().build(accepted[0])
    assert any(k.startswith("tab.") for k in row)
    assert any(k.startswith("img.") for k in row)
    assert any(k.startswith("tmp.") for k in row)


def test_registry_respects_prefixes_off(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    config = FeatureEngineeringConfig(prefixes=False)
    row = FeatureBuilderRegistry(config).build(accepted[0])
    assert any(k == "crop" for k in row)
    assert not any(k.startswith("tab.") for k in row)


def test_build_frame_rectangular(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    frame = build_feature_frame(accepted)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(accepted)
    assert "tab.crop" in frame.columns
    assert "img.pair_count" in frame.columns


def test_build_frame_from_corpus(corpus):
    frame = build_feature_frame(corpus)
    assert len(frame) == len(corpus.accepted())
    assert "tab.crop" in frame.columns


def test_build_frame_empty_raises():
    with pytest.raises(FeatureFrameError):
        build_feature_frame([])


def test_build_features_convenience(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    row = build_features(accepted[0])
    assert "tab.crop" in row
