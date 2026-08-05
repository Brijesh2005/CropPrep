"""GradCAM tests: all CAM methods produce spatial heatmaps + exports."""

from __future__ import annotations

import numpy as np
import pytest

from training.explainability import ImageExplainer, compute_cam
from training.explainability.config import ExplainabilityConfig
from training.explainability.exceptions import CamError


def test_gradcam_heatmap_shapes(full_model, full_sample):
    explainer = ImageExplainer(full_model, ExplainabilityConfig().cam)
    result = explainer.explain(full_sample, index="ndvi", kind="crop")
    assert result["heatmaps"].ndim == 3  # [T, H, W]
    assert result["heatmaps"].shape[1:] == (32, 32)
    assert np.all(result["heatmaps"] >= 0) and np.all(result["heatmaps"] <= 1)


def test_gradcam_evi_yield(full_model, full_sample):
    explainer = ImageExplainer(full_model, ExplainabilityConfig().cam)
    result = explainer.explain(full_sample, index="evi", kind="yield")
    assert result["heatmaps"].shape[1:] == (32, 32)


@pytest.mark.parametrize("method", ["gradcam", "gradcam++", "eigencam", "layercam"])
def test_all_cam_methods(full_model, full_sample, method):
    config = ExplainabilityConfig(cam={"method": method})
    explainer = ImageExplainer(full_model, config.cam)
    result = explainer.explain(full_sample, index="ndvi", kind="crop")
    assert result["heatmaps"].shape == (4, 32, 32)


def test_cam_unknown_method():
    with pytest.raises(CamError):
        compute_cam(torch_zeros(), torch_zeros(), method="nope")


def test_overlay_and_exports(full_model, full_sample, tmp_path):
    explainer = ImageExplainer(full_model, ExplainabilityConfig().cam)
    result = explainer.explain(full_sample, index="ndvi", kind="crop")
    overlay = explainer.overlay(full_sample, result, index="ndvi")
    assert overlay.shape == (32, 32, 3)
    png = explainer.export_png(result, tmp_path / "cam.png")
    npy = explainer.export_numpy(result, tmp_path / "cam.npy")
    assert png.exists() and png.stat().st_size > 0
    assert npy.exists()
    loaded = np.load(npy)
    assert loaded.shape == result["heatmaps"].shape


def test_tabular_only_raises(tabular_model, sample):
    with pytest.raises(CamError):
        ImageExplainer(tabular_model)


def torch_zeros():
    import torch

    return torch.zeros(2, 8, 4, 4)
