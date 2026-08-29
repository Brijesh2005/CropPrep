"""R5.3 numeric-stability regression gates.

Mirrors the ``validation_numerics_probe`` gates that diagnosed ``TR-VAL-001``
(NaN/Inf in validation loss, first detected at
``ndvi_encoder.backbone.blocks.4.2.bn2.drop`` on the P100): fp32-vs-fp16
validation, NDVI/EVI branch stability, temporal-mask semantics, the frozen
split contract and the lightweight train + validation epoch that must stay
finite. These run on CPU with a small timm backbone; the fp16 reproducibility
of the Inf itself is CUDA-only (real 16x8 imagery) and skipped here — it is
covered by the Kaggle probe.
"""

from __future__ import annotations

import importlib.util
import math
from itertools import islice
from pathlib import Path

import pytest
import torch

from training.training import (
    MultiTaskLoss,
    TrainingRunError,
    Validator,
    ValidationError,
    apply_gradient_checkpointing,
)
from training.training.config import ValidationConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEVICE = torch.device("cpu")


@pytest.fixture
def model(full_config):
    from training.models import ModelFactory

    return ModelFactory.create(full_config).to(_DEVICE)


@pytest.fixture
def loss_module(train_config):
    return MultiTaskLoss(train_config.loss)


def _multimodal_loader(batch_size: int = 8):
    """Phase-4-style batches for ``small_full_config`` (ordinal tabular).

    ``feature_dim`` is 4 = 3 numeric + 1 ordinal-encoded categorical column
    (cardinality 2), with the ordinal column set to valid codes.
    """
    class _Loader:
        def __init__(self) -> None:
            self.batches = []
            for _ in range(2):
                tabular = torch.randn(batch_size, 4)
                tabular[:, 3] = torch.randint(0, 2, (batch_size,))
                self.batches.append(
                    {
                        "tabular": tabular,
                        "crop_label": torch.randint(0, 3, (batch_size,)),
                        "yield_label": torch.randn(batch_size, 1),
                        "ndvi": torch.rand(batch_size, 2, 1, 32, 32),
                        "evi": torch.rand(batch_size, 2, 1, 32, 32),
                        "temporal_mask": torch.ones(batch_size, 2, dtype=torch.bool),
                    }
                )

        def __len__(self) -> int:
            return len(self.batches)

        def __iter__(self):
            return iter(self.batches)

    return _Loader()


def _assert_finite(t: torch.Tensor, name: str) -> None:
    assert bool(torch.isfinite(t).all()), f"non-finite tensor in {name}: {t.float().abs().max().item()}"


def _inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: batch[k] for k in ("tabular", "ndvi", "evi", "temporal_mask")
            if k in batch}


def test_forward_all_modalities_finite(model):
    model.eval()
    with torch.no_grad():
        out = model(_inputs(next(iter(_multimodal_loader()))))
    _assert_finite(out.crop_logits, "crop_logits")
    _assert_finite(out.yield_pred, "yield_pred")
    _assert_finite(out.shared_representation, "shared_representation")


def test_backward_all_grads_finite(model, loss_module):
    model.train()
    batch = next(iter(_multimodal_loader()))
    out = model(_inputs(batch))
    loss, _ = loss_module(
        {"crop": out.crop_logits, "yield": out.yield_pred},
        {"crop": batch["crop_label"], "yield": batch["yield_label"]},
    )
    loss.backward()
    assert math.isfinite(loss.item())
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            _assert_finite(p.grad, f"grad[{name}]")


def test_validation_fp32_default_finite(model, loss_module):
    validator = Validator(model, loss_module, device=_DEVICE)
    result = validator.validate(_multimodal_loader())
    assert validator.amp is False
    assert math.isfinite(result.val_loss)
    assert result.samples == 16


def test_fp16_validation_auto_disabled_without_cuda(model, loss_module):
    validator = Validator(
        model, loss_module, device=_DEVICE, amp=True, amp_dtype="float16"
    )
    assert validator.amp == torch.cuda.is_available()
    config = ValidationConfig(amp=True, amp_dtype="float16")
    assert config.amp is True
    assert config.amp_dtype == "float16"
    if not torch.cuda.is_available():
        result = validator.validate(_multimodal_loader())
        assert math.isfinite(result.val_loss)


class _PoisoningLoss(torch.nn.Module):
    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base
        self.tasks = base.tasks
        self.poison = False

    def forward(self, out_dict, targets):
        loss, per_task = self.base(out_dict, targets)
        if self.poison:
            per_task = dict(per_task)
            per_task["crop"] = per_task["crop"] * float("nan")
            loss = torch.stack(list(per_task.values())).sum()
        return loss, per_task


def test_nan_loss_raises_tr_val_001(model, loss_module):
    poison = _PoisoningLoss(loss_module)
    poison.poison = True
    validator = Validator(model, poison, device=_DEVICE)
    with pytest.raises(ValidationError) as excinfo:
        validator.validate(_multimodal_loader())
    error = excinfo.value
    assert error.code == "TR-VAL-001"
    detail = getattr(error, "detail", {}) or {}
    assert "val_loss" in detail.get("metrics", {})
    assert not math.isfinite(detail["metrics"]["val_loss"])
    assert "first_batch" in detail
    assert isinstance(detail["nan_sources"], list)


def test_ndvi_only_and_evi_only_branch_finite(model, loss_module):
    validator = Validator(model, loss_module, device=_DEVICE)
    batch = next(iter(_multimodal_loader()))
    evi_zero = dict(batch, evi=torch.zeros_like(batch["evi"]))
    result_ndvi = validator.validate([evi_zero])
    ndvi_zero = dict(batch, ndvi=torch.zeros_like(batch["ndvi"]))
    result_evi = validator.validate([ndvi_zero])
    assert math.isfinite(result_ndvi.val_loss)
    assert math.isfinite(result_evi.val_loss)


def test_temporal_mask_matches_sequence_and_ignores_padding(model):
    model.eval()
    batch = next(iter(_multimodal_loader()))
    assert batch["temporal_mask"].shape == batch["ndvi"].shape[:2]
    assert batch["temporal_mask"].shape == batch["evi"].shape[:2]

    mask = batch["temporal_mask"].clone()
    mask[:, 1] = False
    garbage = dict(batch, temporal_mask=mask, ndvi=batch["ndvi"].clone())
    garbage["ndvi"][:, 1] = 1e6
    zeroed = dict(batch, temporal_mask=mask.clone())
    zeroed["ndvi"][:, 1] = 0.0

    with torch.no_grad():
        out_garbage = model(_inputs(garbage))
        out_zeroed = model(_inputs(zeroed))
    _assert_finite(out_garbage.crop_logits, "crop_logits(masked garbage)")
    assert torch.allclose(
        out_garbage.crop_logits, out_zeroed.crop_logits, atol=1e-3
    )


def test_frozen_split_manifest_verification():
    csv_path = _REPO_ROOT / "govt_crop_matched_v1" / "crop_supervised_v1.csv"
    manifest_path = (
        _REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"
    )
    if not (csv_path.exists() and manifest_path.exists()):
        pytest.skip("frozen corpus files not present in repo")

    from training.kaggle.frozen_corpus import FrozenCorpusLoader

    manifest = FrozenCorpusLoader(csv_path, manifest_path).validate()
    assert manifest["total_samples"] == 10674
    assert manifest["train_samples"] == 6116
    assert manifest["validation_samples"] == 2267
    assert manifest["test_samples"] == 2291
    assert sum(
        manifest[k]
        for k in ("train_samples", "validation_samples", "test_samples")
    ) == manifest["total_samples"]
    assert manifest["split_strategy"] == "spatial_leave_one_taluk_out"


def test_verify_scripts_importable_from_repo_root():
    for script in (
        "verify_split_composition.py",
        "verify_multimodal_tensors.py",
        "run_pipeline.py",
        "validation_numerics_probe.py",
    ):
        module_name = f"_r5_3_{Path(script).stem}"
        path = _REPO_ROOT / "training" / "kaggle" / "scripts" / script
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(getattr(module, "main", None)), f"{script} lacks main()"


def test_lightweight_train_step_finite(model, loss_module):
    apply_gradient_checkpointing(model, True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    step_loss = float("nan")
    for batch in islice(_multimodal_loader(), 1):
        out = model(_inputs(batch))
        loss, _ = loss_module(
            {"crop": out.crop_logits, "yield": out.yield_pred},
            {"crop": batch["crop_label"], "yield": batch["yield_label"]},
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss = float(loss.detach().item())
    assert math.isfinite(step_loss)
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            _assert_finite(p.grad, f"grad[{name}]")
    for p in model.parameters():
        _assert_finite(p, "param")


def test_lightweight_validation_epoch_finite(model, loss_module):
    apply_gradient_checkpointing(model, True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for batch in islice(_multimodal_loader(), 1):
        out = model(_inputs(batch))
        loss, _ = loss_module(
            {"crop": out.crop_logits, "yield": out.yield_pred},
            {"crop": batch["crop_label"], "yield": batch["yield_label"]},
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    validator = Validator(model, loss_module, device=_DEVICE, amp=False)
    result = validator.validate(_multimodal_loader())
    assert math.isfinite(result.val_loss)
    assert result.samples > 0


def test_gradcheckpointed_multistep_finite(model, loss_module):
    apply_gradient_checkpointing(model, True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for batch in islice(_multimodal_loader(), 2):
        out = model(_inputs(batch))
        loss, _ = loss_module(
            {"crop": out.crop_logits, "yield": out.yield_pred},
            {"crop": batch["crop_label"], "yield": batch["yield_label"]},
        )
        optimizer.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                _assert_finite(p.grad, f"grad[{name}]")
        optimizer.step()
        losses.append(float(loss.detach().item()))
    assert all(math.isfinite(value) for value in losses)
    assert model.use_image


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fp16-vs-fp32 validation contrast needs CUDA"
)
def test_fp16_vs_fp32_validation_contrast_finite(model, loss_module):
    validator_fp32 = Validator(model, loss_module, device="cuda", amp=False)
    validator_fp16 = Validator(
        model, loss_module, device="cuda", amp=True, amp_dtype="float16"
    )
    result_fp32 = validator_fp32.validate(_multimodal_loader().batches)
    result_fp16 = validator_fp16.validate(_multimodal_loader().batches)
    assert math.isfinite(result_fp32.val_loss)
    assert math.isfinite(result_fp16.val_loss)