"""Tests for the temporal pipeline."""

from __future__ import annotations

from datetime import date

import torch

from ai.preprocessing.config import TemporalConfig
from ai.preprocessing.exceptions import FitError, SampleRejectedError
from ai.preprocessing.temporal_pipeline import TemporalPipeline


def _tensor(value: float = 1.0):
    return torch.full((1, 2, 2), value, dtype=torch.float32)


def _dates(*days):
    return [date(2020, 7, day) for day in days]


def test_pad_to_max_with_mask():
    pipeline = TemporalPipeline(TemporalConfig(max_observations=4)).fit([])
    ndvi = [_tensor(1.0), _tensor(1.0), _tensor(1.0)]
    evi = [_tensor(1.0)] * 3
    seq_n, seq_e, mask = pipeline.transform_sequence(ndvi, evi, _dates(1, 8, 15))
    assert seq_n.shape == (4, 1, 2, 2)
    assert mask.shape == (4,)
    assert mask[:3].sum() == 3  # 3 real, 1 padded
    assert seq_n[3].abs().sum() == 0  # padded position is zero


def test_truncation_tail():
    pipeline = TemporalPipeline(TemporalConfig(max_observations=2)).fit([])
    ndvi = [_tensor()] * 5
    evi = [_tensor()] * 5
    seq_n, _, _ = pipeline.transform_sequence(ndvi, evi, _dates(1, 2, 3, 4, 5))
    assert seq_n.shape[0] == 2
    assert seq_n[0, 0, 0, 0] == 1.0  # first two kept (tail truncation)


def test_sort_out_of_order():
    pipeline = TemporalPipeline(TemporalConfig(max_observations=4)).fit([])
    # ndvi tensors are indexed by date position: 15 -> 1.0, 1 -> 2.0, 8 -> 3.0.
    ndvi = [_tensor(float(i)) for i in (1, 2, 3)]
    seq_n, _, _ = pipeline.transform_sequence(ndvi, [_tensor()] * 3, _dates(15, 1, 8))
    # After sorting, the earliest date (1st of month) carries tensor value 2.0.
    assert seq_n[0, 0, 0, 0] == 2.0


def test_duplicate_dates_dropped():
    pipeline = TemporalPipeline(TemporalConfig(max_observations=4)).fit([])
    ndvi = [_tensor(float(i)) for i in (1, 2)]
    seq_n, _, _ = pipeline.transform_sequence(ndvi, [_tensor()] * 2, _dates(1, 1))
    assert (seq_n[:, 0, 0, 0] == 1.0).sum() == 1  # only one kept


def test_too_few_observations_rejected():
    pipeline = TemporalPipeline(TemporalConfig(min_observations=3)).fit([])
    import pytest

    with pytest.raises(SampleRejectedError):
        pipeline.transform_sequence([_tensor()], [_tensor()], _dates(1))


def test_missing_ndvi_zero_filled():
    pipeline = TemporalPipeline(TemporalConfig(max_observations=2)).fit([])
    seq_n, seq_e, _ = pipeline.transform_sequence([None], [_tensor()], _dates(1))
    assert seq_n[0].abs().sum() == 0
    assert seq_e[0, 0, 0, 0] == 1.0


def test_left_padding():
    pipeline = TemporalPipeline(TemporalConfig(max_observations=3, pad_mode="left")).fit([])
    seq_n, _, mask = pipeline.transform_sequence([_tensor()], [_tensor()], _dates(1))
    assert mask[2] == 1.0  # real observation at the end


def test_unfitted_raises():
    pipeline = TemporalPipeline()
    import pytest

    with pytest.raises(FitError):
        pipeline.transform_sequence([_tensor()], [_tensor()], _dates(1))


def test_save_load(tmp_path):
    pipeline = TemporalPipeline(TemporalConfig(max_observations=4)).fit([])
    out = pipeline.save(tmp_path)
    loaded = TemporalPipeline.load(out)
    assert loaded.config.max_observations == 4
