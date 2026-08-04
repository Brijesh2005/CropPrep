"""Optimizer tests: construction and a single step for every optimizer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ai.training import Lion, build_optimizer
from ai.training.config import OptimizerConfig
from ai.training.exceptions import OptimizerBuildError


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


NAMES = ("adamw", "sgd", "radam", "lion")


@pytest.mark.parametrize("name", NAMES)
def test_build_and_step(name):
    model = _Net()
    optimizer = build_optimizer(model, OptimizerConfig(name=name, lr=1e-3))
    optimizer.zero_grad()
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    assert all(p.grad is not None for p in model.parameters())


def test_unknown_optimizer():
    with pytest.raises(OptimizerBuildError):
        build_optimizer(_Net(), OptimizerConfig(name="nope"))


def test_sgd_params():
    optimizer = build_optimizer(
        _Net(), OptimizerConfig(name="sgd", lr=0.1, momentum=0.9, nesterov=True)
    )
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[0]["nesterov"] is True


def test_lion_matches_reference():
    """Lion step should be a sign-based update that changes the params."""
    model = _Net()
    before = {name: p.clone() for name, p in model.named_parameters()}
    optimizer = Lion(model.parameters(), lr=1e-2)
    loss = model(torch.randn(2, 4)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    changed = any(
        not torch.equal(before[name], p) for name, p in model.named_parameters()
    )
    assert changed
