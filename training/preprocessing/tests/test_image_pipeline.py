"""Tests for the image pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from training.preprocessing.config import ImageConfig
from training.preprocessing.exceptions import FitError
from training.preprocessing.image_pipeline import ImagePipeline


def _patch(fill=0.5, size=32):
    return np.full((size, size), fill, dtype="float32")


def test_minmax_ndvi_scaling():
    pipeline = ImagePipeline(ImageConfig(size=32, normalize="minmax")).fit([])
    tensor = pipeline.transform_patch(_patch(fill=0.0), "NDVI")
    assert tensor.shape == (1, 32, 32)
    assert float(tensor[0, 0, 0]) == pytest.approx(0.5)  # (-1..1) -> (0..1)


def test_minmax_evi_uses_evi_range():
    pipeline = ImagePipeline(ImageConfig(size=32, normalize="minmax")).fit([])
    tensor = pipeline.transform_patch(_patch(fill=1.0), "EVI")
    assert float(tensor[0, 0, 0]) == pytest.approx(1.0)


def test_nan_handling_zero():
    pipeline = ImagePipeline(ImageConfig(size=32, normalize="identity")).fit([])
    patch = _patch()
    patch[0, 0] = np.nan
    tensor = pipeline.transform_patch(patch, "NDVI")
    assert float(tensor[0, 0, 0]) == pytest.approx(0.0)


def test_invalid_pixel_zeroed():
    pipeline = ImagePipeline(ImageConfig(size=32, normalize="identity")).fit([])
    mask = np.ones((32, 32), dtype=bool)
    mask[0, 0] = False
    tensor = pipeline.transform_patch(_patch(fill=0.9), "NDVI", mask=mask)
    assert float(tensor[0, 0, 0]) == pytest.approx(0.0)


def test_resize_smaller_patch():
    pipeline = ImagePipeline(ImageConfig(size=16, normalize="identity")).fit([])
    tensor = pipeline.transform_patch(_patch(size=8), "NDVI")
    assert tensor.shape == (1, 16, 16)


def test_standard_normalization_requires_extractor():
    pipeline = ImagePipeline(ImageConfig(size=32, normalize="standard"))
    with pytest.raises(FitError):
        pipeline.transform_patch(_patch(), "NDVI")  # not fitted
    with pytest.raises(Exception):
        pipeline.fit([], extractor=None)  # standard needs extractor


def test_clip_to_range():
    pipeline = ImagePipeline(ImageConfig(size=32, normalize="minmax", clip=True)).fit([])
    # NDVI values outside [-1,1] are clipped then mapped to [0,1].
    tensor = pipeline.transform_patch(_patch(fill=5.0), "NDVI")
    assert float(tensor[0, 0, 0]) == pytest.approx(1.0)


def test_patch_with_3d_input():
    pipeline = ImagePipeline(ImageConfig(size=8, normalize="identity")).fit([])
    tensor = pipeline.transform_patch(np.ones((1, 8, 8)), "NDVI")
    assert tensor.shape == (1, 8, 8)


def test_summary_and_persistence(tmp_path):
    pipeline = ImagePipeline(ImageConfig(size=8, normalize="minmax")).fit([])
    out = pipeline.save(tmp_path)
    loaded = ImagePipeline.load(out)
    assert loaded.config.size == 8
    t1 = pipeline.transform_patch(_patch(size=8), "NDVI")
    t2 = loaded.transform_patch(_patch(size=8), "NDVI")
    assert np.allclose(t1.numpy(), t2.numpy())
