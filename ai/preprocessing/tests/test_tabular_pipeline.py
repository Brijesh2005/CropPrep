"""Tests for the tabular pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ai.preprocessing.config import TabularConfig
from ai.preprocessing.exceptions import FitError
from ai.preprocessing.tabular_pipeline import TabularPipeline


def _fields(village="A", rainfall=2100.0, crop="Rice", district="DK", year=2020):
    from services.spatial_alignment.observation import TabularFeatures

    return {
        "village": village,
        "district": district,
        "crop": crop,
        "rainfall_mm": rainfall,
        "year": year,
    }


class _Obs:
    def __init__(self, fields):
        from services.spatial_alignment.observation import TabularFeatures

        self.tabular = TabularFeatures(crop=fields.get("crop"),
                                       yield_value=5200.0, fields=fields,
                                       matched_level="village")


@pytest.fixture
def samples():
    return [
        _Obs(_fields(rainfall=2100)),
        _Obs(_fields(rainfall=2200)),
        _Obs(_fields(rainfall=2300)),
    ]


def test_fit_and_transform_shape(samples):
    config = TabularConfig(
        scaler="standard", categorical_encoding="onehot",
        numeric_features=["rainfall_mm"], categorical_features=["village", "district"],
        exclude_columns=["crop", "year"],
    )
    pipeline = TabularPipeline(config).fit(samples)
    tensor = pipeline.transform(samples[0])
    assert tensor.dim() == 1
    assert tensor.shape[0] == len(pipeline.feature_names)
    assert pipeline.feature_names[0] == "rainfall_mm"


def test_auto_inference_and_constant_drop(samples):
    config = TabularConfig(scaler="standard", categorical_encoding="ordinal",
                           exclude_columns=["crop", "year"])
    pipeline = TabularPipeline(config).fit(samples)
    # district is constant -> dropped from numeric? it's categorical; rainfall stays.
    assert "rainfall_mm" in pipeline.numeric_features
    assert pipeline.fitted is True


def test_constant_numeric_dropped():
    config = TabularConfig(exclude_columns=["crop", "year"])
    samples = [
        _Obs({**_fields(), "rainfall_mm": 2100}),
        _Obs({**_fields(), "rainfall_mm": 2100}),
    ]
    pipeline = TabularPipeline(config).fit(samples)
    assert "rainfall_mm" in pipeline.dropped_constant


def test_missing_value_filled():
    config = TabularConfig(scaler="none", categorical_encoding="none",
                           numeric_features=["rainfall_mm"], categorical_features=[],
                           exclude_columns=["crop", "year"])
    samples = [
        _Obs({**_fields(), "rainfall_mm": None}),
        _Obs({**_fields(), "rainfall_mm": 3000}),
        _Obs({**_fields(), "rainfall_mm": 2000}),
    ]
    pipeline = TabularPipeline(config).fit(samples)
    tensor = pipeline.transform(samples[0])
    assert tensor.shape[0] == 1
    assert float(tensor[0]) == pytest.approx(2500.0)  # mean fill of [3000, 2000]


def test_unfitted_raises(samples):
    pipeline = TabularPipeline()
    with pytest.raises(FitError):
        pipeline.transform(samples[0])


def test_validate_reports_missing_column(samples):
    config = TabularConfig(numeric_features=["missing_col"], categorical_features=[])
    pipeline = TabularPipeline(config).fit(samples)
    issues = pipeline.validate(samples[0])
    assert any("missing_col" in issue for issue in issues)


def test_save_load_roundtrip(samples, tmp_path):
    pipeline = TabularPipeline(
        TabularConfig(scaler="standard", categorical_encoding="onehot",
                      numeric_features=["rainfall_mm"],
                      categorical_features=["village", "district"],
                      exclude_columns=["crop", "year"])
    ).fit(samples)
    out = pipeline.save(tmp_path)
    loaded = TabularPipeline.load(out)
    assert loaded.feature_names == pipeline.feature_names
    t1 = pipeline.transform(samples[0])
    t2 = loaded.transform(samples[0])
    assert np.allclose(t1.numpy(), t2.numpy())
