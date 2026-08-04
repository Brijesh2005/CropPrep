"""Optimizer factory for the training engine.

Supports AdamW, SGD, RAdam and Lion. Lion (Chen et al., 2023) is provided
as a self-contained implementation so it works without extra dependencies.

``build_optimizer`` constructs the optimizer from a
:class:`~ai.training.config.OptimizerConfig` and the model's trainable
parameters.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import torch
from torch import nn

from .config import OptimizerConfig
from .exceptions import OptimizerBuildError


class Lion(torch.optim.Optimizer):
    """Self-contained Lion optimizer (Chen et al., 2023).

    Lion applies a sign-based update with separate moments:

    .. math::

        c_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t
        m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t
        \\theta_t = \\theta_{t-1} - \\text{lr} \\cdot
            \\text{sign}(c_t)

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate.
        betas: ``(beta1, beta2)`` moments coefficients.
        weight_decay: L2 weight decay.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if group["weight_decay"] != 0.0:
                    grad = grad.add(param, alpha=group["weight_decay"])
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]

                update = exp_avg.lerp(grad, 1 - beta1)
                exp_avg.lerp_(grad, 1 - beta2)
                param.add_(torch.sign(update), alpha=-group["lr"])
        return loss


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def _trainable_params(model: nn.Module) -> Iterator[nn.Parameter]:
    for param in model.parameters():
        if param.requires_grad:
            yield param


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig | None = None,
    *,
    params: Iterable[nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer from a config for ``model``'s trainable parameters.

    Args:
        model: The model (or ``nn.Module`` container of trainable params).
        config: Validated :class:`OptimizerConfig`.
        params: Explicit parameter iterable (overrides ``model.parameters()``).

    Raises:
        OptimizerBuildError: On an unknown optimizer name.
    """
    config = config or OptimizerConfig()
    param_iter = params if params is not None else _trainable_params(model)
    name = (config.name or "adamw").strip().lower()

    if name == "adamw":
        return torch.optim.AdamW(
            param_iter, lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay, eps=config.eps,
        )
    if name == "sgd":
        return torch.optim.SGD(
            param_iter, lr=config.lr, momentum=config.momentum,
            weight_decay=config.weight_decay, nesterov=config.nesterov,
        )
    if name == "radam":
        return torch.optim.RAdam(
            param_iter, lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay, eps=config.eps,
        )
    if name == "lion":
        return Lion(
            param_iter, lr=config.lr, betas=(config.lion_beta1, config.lion_beta2),
            weight_decay=config.weight_decay,
        )
    raise OptimizerBuildError(f"unknown optimizer {config.name!r}")
