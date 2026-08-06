"""Tests for the image / sequence feature builder."""

from __future__ import annotations

import numpy as np
import pytest

from training.feature_engineering.config import ImageFeatureConfig
from training.feature_engineering.exceptions import MissingExtractorError
from training.feature_engineering.image import ImageFeatureBuilder


def test_metadata_features(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    row = ImageFeatureBuilder().build(accepted[0])
    assert "img.ndvi_count" in row
    assert "img.evi_count" in row
    assert "img.pair_count" in row
    assert row["img.pair_count"] == len(accepted[0].sequence.pairs)
    assert "img.resolution" in row
    assert "img.max_gap_days" in row


def test_patch_stats_require_extractor(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    builder = ImageFeatureBuilder(ImageFeatureConfig(extract_patch_stats=True))
    with pytest.raises(MissingExtractorError):
        builder.build(accepted[0])


def test_patch_stats_with_extractor(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    builder = ImageFeatureBuilder(
        ImageFeatureConfig(extract_patch_stats=True, max_dates=2, patch_size=16)
    )

    class _Patch:
        array = np.ones((16, 16), dtype="float32")
        mask = np.ones((16, 16), dtype=bool)

    def extractor(path, lon, lat, size=16):
        return _Patch()

    row = builder.build(accepted[0], extractor=extractor)
    assert "img.d0.ndvi.mean" in row
    assert row["img.d0.ndvi.mean"] == 1.0
    assert row["img.d0.evi.max"] == 1.0
    assert row["img.patch_dates_used"] >= 1


def test_max_dates_caps_patch_features(accepted):
    if not accepted:
        pytest.skip("no accepted observations in fixture corpus")
    builder = ImageFeatureBuilder(
        ImageFeatureConfig(extract_patch_stats=True, max_dates=1, patch_size=16)
    )

    class _Patch:
        array = np.zeros((16, 16), dtype="float32")
        mask = np.ones((16, 16), dtype=bool)

    def extractor(path, lon, lat, size=16):
        return _Patch()

    row = builder.build(accepted[0], extractor=extractor)
    assert row["img.patch_dates_used"] <= 1
    assert "img.d1.ndvi.mean" not in row
