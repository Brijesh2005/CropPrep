"""Tests for the label pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from training.preprocessing.config import LabelConfig
from training.preprocessing.label_pipeline import LabelPipeline


class _Obs:
    def __init__(self, crop="Rice", yield_value=5200.0):
        self.crop = crop
        self.yield_value = yield_value


def test_crop_and_yield_encode():
    pipeline = LabelPipeline(LabelConfig(yield_scaler="standard")).fit(
        [_Obs("Rice", 5000), _Obs("Coconut", 3100), _Obs("Rice", 5400)]
    )
    crop, yield_value = pipeline.transform(_Obs("Rice", 5200))
    assert int(crop) == 0  # Rice -> 0
    assert crop.dtype == torch_int64()
    assert yield_value.dim() == 0
    assert pipeline.num_classes == 2


def test_yield_scaled_mean_zero():
    pipeline = LabelPipeline(LabelConfig(yield_scaler="standard")).fit(
        [_Obs("Rice", 5000), _Obs("Rice", 6000)]
    )
    _, yield_value = pipeline.transform(_Obs("Rice", 5500))
    assert float(yield_value) == pytest.approx(0.0, abs=1e-4)


def test_yield_scaler_none():
    pipeline = LabelPipeline(LabelConfig(yield_scaler="none")).fit(
        [_Obs("Rice", 5200)]
    )
    _, yield_value = pipeline.transform(_Obs("Rice", 5200))
    assert float(yield_value) == pytest.approx(5200.0)


def test_onehot_crop_encoding():
    pipeline = LabelPipeline(LabelConfig(crop_encoding="onehot")).fit(
        [_Obs("Rice"), _Obs("Coconut")]
    )
    crop, _ = pipeline.transform(_Obs("Rice"))
    assert crop.shape[0] == 2
    assert crop[0] == 1.0


def test_inverse_crop():
    pipeline = LabelPipeline().fit([_Obs("Rice"), _Obs("Coconut")])
    assert pipeline.inverse_crop([0, 1]) == ["Rice", "Coconut"]


def test_save_load(tmp_path):
    pipeline = LabelPipeline().fit([_Obs("Rice"), _Obs("Coconut")])
    out = pipeline.save(tmp_path)
    loaded = LabelPipeline.load(out)
    assert loaded.num_classes == 2
    assert loaded.inverse_crop([1]) == ["Coconut"]


def test_yield_scale_diagnostics_flag_mixed_units():
    """R5.2 Task 5/7: a mixed-unit yield target must be flagged at fit time."""
    pipeline = LabelPipeline(LabelConfig(yield_scaler="standard")).fit(
        [_Obs("Coconut", 62), _Obs("Coconut", 73730), _Obs("Coconut", 0.53),
         _Obs("Coconut", 1.5)]
    )
    stats = pipeline.yield_scale_stats
    assert stats is not None
    assert stats["dynamic_range_ratio"] > 1e3
    assert any("mixed units" in w for w in pipeline.warnings)
    # scaler mean is dominated by the kg/ha scale
    assert pipeline.yield_scaler.mean_[0] > 100
    assert "yield_scale_stats" in pipeline.summary()
    assert "warnings" in pipeline.summary()


def test_yield_scale_diagnostics_no_warning_clean_target():
    pipeline = LabelPipeline(LabelConfig(yield_scaler="standard")).fit(
        [_Obs("Coconut", 5000), _Obs("Coconut", 6000), _Obs("Coconut", 5500)]
    )
    assert pipeline.yield_scale_stats is not None
    assert pipeline.warnings == []
    assert pipeline.yield_scale_stats["dynamic_range_ratio"] < 10


def test_yield_scale_diagnostics_constant_collapse():
    """A target whose scaled values collapse to a single point must be flagged.

    Mirrors the real defect: 201 district samples all scale to -0.3877.
    """
    pipeline = LabelPipeline(LabelConfig(yield_scaler="standard")).fit(
        [_Obs("Coconut", 1.0), _Obs("Coconut", 1.0), _Obs("Coconut", 1.0)]
    )
    assert pipeline.yield_scale_stats is not None
    assert pipeline.yield_scale_stats["scaled_distinct_values"] <= 1
    assert any("collapse" in w for w in pipeline.warnings)


def torch_int64():
    import torch

    return torch.int64
