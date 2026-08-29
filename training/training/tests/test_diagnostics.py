"""Diagnostics tests: batch profiling, image-shape assertion, NaN tracing.

R5.2-mandated observability: the first-batch multimodal diagnostic must prove
the exact tensor contract (B / T / C / H / W, real-vs-zero-filled, finiteness)
that reaches the model, and the shape assertion must fail loudly (never be
suppressed) when the contract is violated.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from torch import nn

from training.models import NdviEncoder
from training.training.diagnostics import (
    assert_image_batch_shape,
    nan_source_hooks,
    profile_batch,
    tensor_stats,
)
from training.training.utils import apply_gradient_checkpointing


def _multimodal_batch(
    n: int = 2,
    t: int = 3,
    hw: int = 32,
    *,
    zero_fill: bool = False,
    fill_frames: int = 1,
    mask_all_padding: bool = False,
) -> dict[str, Any]:
    """A Phase-4-shaped multimodal batch (optionally with zero-filled time steps)."""
    ndvi = torch.rand(n, t, 1, hw, hw)
    evi = torch.rand(n, t, 1, hw, hw)
    mask = torch.ones(n, t, dtype=torch.bool)
    if zero_fill:
        for i in range(n):
            for k in range(fill_frames):
                ndvi[i, k] = 0.0
                evi[i, k] = 0.0
                mask[i, k] = bool(mask_all_padding)
    if mask_all_padding:
        mask.fill_(0)
    return {
        "tabular": torch.rand(n, 4),
        "ndvi": ndvi,
        "evi": evi,
        "temporal_mask": mask,
        "crop_label": torch.randint(0, 3, (n,)),
        "yield_label": torch.rand(n, 1),
    }


# --------------------------------------------------------------------------- #
# profile_batch
# --------------------------------------------------------------------------- #


def test_profile_batch_captures_tensor_contract():
    batch = _multimodal_batch(n=2, t=3, hw=32)
    profile = profile_batch(batch)

    assert profile["ndvi_frames"]["shape"] == [2, 3, 1, 32, 32]
    assert profile["ndvi_frames"]["real"] == 6          # 2 * 3
    assert profile["ndvi_frames"]["zero_filled"] == 0
    assert profile["batch_size"] == 2
    assert profile["tabular"]["finite"] is True
    assert profile["mask"]["ones"] == 6


def test_profile_batch_counts_zero_filled_frames():
    batch = _multimodal_batch(n=2, t=3, zero_fill=True, fill_frames=1)
    profile = profile_batch(batch)
    assert profile["ndvi_frames"]["real"] == 4          # 2 * (3 - 1)
    assert profile["ndvi_frames"]["zero_filled"] == 2
    assert profile["evi_frames"]["zero_filled"] == 2


def test_profile_batch_is_json_serialisable():
    batch = _multimodal_batch()
    json.dumps(profile_batch(batch))


def test_tensor_stats_reports_nan_and_inf():
    stats = tensor_stats(torch.tensor([1.0, float("nan"), float("inf"), -2.0]))
    assert stats["nan"] == 1
    assert stats["inf"] == 1
    assert stats["finite"] is False


# --------------------------------------------------------------------------- #
# assert_image_batch_shape — mandated pre-feed contract
# --------------------------------------------------------------------------- #


def test_assert_passes_on_correct_batch():
    batch = _multimodal_batch(n=2, t=3, hw=32)
    profile = assert_image_batch_shape(batch, expected_hw=32)
    assert profile["image_assert_passed"] is True
    assert profile["expected_hw"] == 32


def test_assert_accepts_tabular_only():
    profile = assert_image_batch_shape({"tabular": torch.rand(4, 5)}, expected_hw=None)
    assert profile["image_assert_passed"] is True


def test_assert_rejects_wrong_channel():
    batch = _multimodal_batch()
    batch["ndvi"] = torch.rand(2, 3, 3, 32, 32)
    with pytest.raises(AssertionError, match="single-channel"):
        assert_image_batch_shape(batch, expected_hw=32)


def test_assert_rejects_wrong_spatial_size():
    batch = _multimodal_batch(hw=16)
    with pytest.raises(AssertionError, match="224"):
        assert_image_batch_shape(batch, expected_hw=224)


def test_assert_rejects_empty_sequence():
    batch = _multimodal_batch()
    batch["ndvi"] = torch.rand(2, 0, 1, 32, 32)
    batch["evi"] = torch.rand(2, 0, 1, 32, 32)
    batch["temporal_mask"] = torch.ones(2, 0, dtype=torch.bool)
    with pytest.raises(AssertionError, match="T=0"):
        assert_image_batch_shape(batch, expected_hw=32)


def test_assert_rejects_non_finite_inputs():
    batch = _multimodal_batch()
    batch["ndvi"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        assert_image_batch_shape(batch, expected_hw=32)


def test_assert_rejects_missing_mask():
    batch = _multimodal_batch()
    del batch["temporal_mask"]
    with pytest.raises(AssertionError, match="temporal_mask"):
        assert_image_batch_shape(batch, expected_hw=32)


def test_assert_rejects_all_padding_mask():
    batch = _multimodal_batch(mask_all_padding=True)
    with pytest.raises(AssertionError, match="no real timesteps"):
        assert_image_batch_shape(batch, expected_hw=32)


def test_assert_uses_custom_error_type_and_detail():
    class _CustomError(Exception):
        pass

    batch = _multimodal_batch()
    batch["evi"] = torch.rand(2, 3, 1, 64, 64)
    with pytest.raises(_CustomError, match="verify context"):
        assert_image_batch_shape(
            batch, 32, error_type=_CustomError, detail="verify context"
        )


# --------------------------------------------------------------------------- #
# nan_source_hooks — first non-finite module output attribution
# --------------------------------------------------------------------------- #


def test_nan_source_hooks_records_first_non_finite_module():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    with torch.no_grad():
        model[0].weight[0, 0] = float("nan")
    model.eval()

    with nan_source_hooks(model) as sources:
        with torch.no_grad():
            model(torch.zeros(3, 4))

    assert len(sources) == 1
    assert sources[0]["module"] == "0"
    assert sources[0]["nan"] > 0
    assert sources[0]["inf"] == 0
    assert sources[0]["shape"] == [3, 4]


def test_nan_source_hooks_clean_model_records_nothing():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model.eval()
    with nan_source_hooks(model) as sources:
        with torch.no_grad():
            model(torch.randn(3, 4))
    assert sources == []


def test_nan_source_hooks_graph_neutral_under_gradient_checkpointing():
    model = _TangledModel()
    apply_gradient_checkpointing(model, True)
    model.train()
    images = torch.rand(2, 2, 1, 32, 32)

    with nan_source_hooks(model) as sources:
        out = model.ndvi_encoder(images)

    assert sources == []
    try:
        out.sum().backward()
    except Exception as exc:  # pragma: no cover - failure assertion path
        pytest.fail(
            "checkpointed backward must be unaffected by NaN hooks, "
            f"got {type(exc).__name__}: {exc}"
        )
    grads = [p.grad for p in model.ndvi_encoder.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all().item() for g in grads)


def test_nan_source_hooks_still_attributes_nan_with_grad_enabled_forward():
    model = _TangledModel()
    with torch.no_grad():
        for tensor in model.ndvi_encoder.parameters():
            if tensor.ndim >= 2:
                tensor[0, 0] = float("nan")
                break
    model.train()

    with nan_source_hooks(model) as sources:
        out = model.ndvi_encoder(torch.rand(2, 1, 1, 32, 32))

    assert len(sources) >= 1
    assert sources[0]["nan"] > 0


# --------------------------------------------------------------------------- #
# apply_gradient_checkpointing — setter-based wiring (OOM fix)
# --------------------------------------------------------------------------- #


class _TangledModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ndvi_encoder = NdviEncoder("mobilenetv3_small_050", input_size=32)


def test_apply_gradient_checkpointing_toggles_matching_setters():
    model = _TangledModel()
    assert model.ndvi_encoder._checkpoint_timesteps is False

    apply_gradient_checkpointing(model, True)
    assert model.ndvi_encoder._checkpoint_timesteps is True
    assert getattr(model.ndvi_encoder, "_cropfusion_checkpointed", False) is False

    apply_gradient_checkpointing(model, False)
    assert model.ndvi_encoder._checkpoint_timesteps is False


def test_apply_gradient_checkpointing_noop_when_disabled():
    model = _TangledModel()
    apply_gradient_checkpointing(model, False)
    assert model.ndvi_encoder._checkpoint_timesteps is False