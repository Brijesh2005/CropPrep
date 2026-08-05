"""Loss tests: MAE, multi-task weighting strategies, GradNorm controller."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.training import GradNormController, MAELoss, MultiTaskLoss
from training.training.config import LossConfig


def _inputs() -> dict[str, torch.Tensor]:
    return {"crop": torch.randn(4, 3), "yield": torch.randn(4, 1)}


def _targets() -> dict[str, torch.Tensor]:
    return {"crop": torch.tensor([0, 1, 2, 1]), "yield": torch.randn(4, 1)}


def test_mae_value():
    loss = MAELoss()
    value = loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0]))
    assert abs(float(value) - 1.0) < 1e-6


def test_multi_task_fixed():
    loss = MultiTaskLoss(LossConfig(weighting_mode="fixed"))
    total, per = loss(_inputs(), _targets())
    assert set(per) == {"crop", "yield"}
    assert torch.isfinite(total)


def test_multi_task_uncertainty_backward():
    loss = MultiTaskLoss(LossConfig(weighting_mode="uncertainty"))
    total, per = loss(_inputs(), _targets())
    total.backward()
    assert loss.log_var_crop.grad is not None
    assert loss.log_var_yield.grad is not None


def test_multi_task_mae_yield():
    loss = MultiTaskLoss(LossConfig(yield_loss="mae"))
    total, per = loss(_inputs(), _targets())
    assert torch.isfinite(total)


class _TinyShared(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_encoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared_encoder(x)
        return {"crop": nn.Linear(8, 3)(h), "yield": nn.Linear(8, 1)(h)}


def test_gradnorm_controller_updates_weights():
    model = _TinyShared()
    loss_module = MultiTaskLoss(LossConfig(weighting_mode="gradnorm"))
    controller = GradNormController(model, loss_module, alpha=1.5)

    inp = {"crop": torch.randn(4, 8), "yield": torch.randn(4, 8)}
    tgt = _targets()
    out = {"crop": model(inp["crop"])["crop"], "yield": model(inp["yield"])["yield"]}

    per_task = loss_module.per_task_losses(out, tgt)
    weights = controller.apply(per_task)
    assert set(weights) == {"crop", "yield"}

    # Main backward still works after the GradNorm update.
    total, _ = loss_module.combine(per_task)
    total.backward()
    assert any(p.grad is not None for p in model.shared_encoder.parameters())

    # Weights renormalize to sum == number of tasks.
    assert abs(sum(weights.values()) - 2.0) < 1e-4


def test_gradnorm_state_round_trip():
    model = _TinyShared()
    loss_module = MultiTaskLoss(LossConfig(weighting_mode="gradnorm"))
    controller = GradNormController(model, loss_module)
    state = controller.state_dict()
    restored = GradNormController(model, loss_module)
    restored.load_state_dict(state)
    assert restored._initial_losses == controller._initial_losses
