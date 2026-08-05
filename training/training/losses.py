"""Losses for the multi-task CropFusion model.

Builds on the Phase 5 loss interfaces (``CrossEntropyLoss``,
``LabelSmoothingLoss``, ``FocalLoss``, ``MSELoss``, ``HuberLoss``) and adds:

* :class:`MAELoss` — mean absolute error for yield regression.
* :class:`MultiTaskLoss` — composes the per-task losses under one of three
  weighting strategies:

  * ``fixed`` — configured constant task weights,
  * ``uncertainty`` — Kendall et al. (2018) Gaussian uncertainty weighting
    (learned per-task log-variances),
  * ``gradnorm`` — GradNorm (Chen et al., 2018) with learned task weights
    updated by the :class:`GradNormController`.

:class:`GradNormController` implements the optional GradNorm algorithm. It is
driven by the trainer once per optimizer step.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from training.models.losses import (
    CrossEntropyLoss,
    FocalLoss,
    HuberLoss,
    LabelSmoothingLoss,
    MSELoss,
)

from .config import LossConfig
from .exceptions import LossBuildError


class MAELoss(nn.Module):
    """Mean absolute error for yield regression."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        pred = inputs
        target = targets
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 2 and target.size(1) == 1:
            target = target.squeeze(1)
        errors = (pred.float() - target.float()).abs()
        if self.reduction == "sum":
            return errors.sum()
        return errors.mean()


# --------------------------------------------------------------------------- #
# Task-loss factory
# --------------------------------------------------------------------------- #


def build_task_loss(name: str, config: LossConfig) -> nn.Module:
    """Build the concrete per-task loss for ``name`` (``crop`` | ``yield``)."""
    if name == "crop":
        crop = config.crop_loss
        if crop == "cross_entropy":
            return CrossEntropyLoss(reduction=config.reduction)
        if crop == "label_smoothing":
            return LabelSmoothingLoss(
                smoothing=config.label_smoothing, reduction=config.reduction
            )
        if crop == "focal":
            return FocalLoss(gamma=config.focal_gamma, reduction=config.reduction)
        raise LossBuildError(f"unknown crop loss {crop!r}")
    if name == "yield":
        yield_loss = config.yield_loss
        if yield_loss == "mse":
            return MSELoss(reduction=config.reduction)
        if yield_loss == "huber":
            return HuberLoss(reduction=config.reduction)
        if yield_loss == "mae":
            return MAELoss(reduction=config.reduction)
        raise LossBuildError(f"unknown yield loss {yield_loss!r}")
    raise LossBuildError(f"no task-loss factory for task {name!r}")


# --------------------------------------------------------------------------- #
# Multi-task loss
# --------------------------------------------------------------------------- #


class MultiTaskLoss(nn.Module):
    """Weighted composition of the per-task losses.

    Args:
        config: Validated :class:`LossConfig`.
        tasks: Optional mapping of task name -> :class:`~ai.models.interfaces.
            TaskLoss`. Built from ``config`` when omitted (``crop`` +
        ``yield``).

    Forward returns ``(total_loss, per_task_losses)``. In ``uncertainty`` mode
    the per-task log-variances are learned alongside the model; in
    ``gradnorm`` mode the task weights are trainable and updated by
    :class:`GradNormController`.
    """

    def __init__(
        self,
        config: LossConfig | None = None,
        tasks: Mapping[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LossConfig()
        if tasks is None:
            tasks = {
                "crop": build_task_loss("crop", self.config),
                "yield": build_task_loss("yield", self.config),
            }
        if not tasks:
            raise LossBuildError("MultiTaskLoss requires at least one task")
        self.tasks = nn.ModuleDict(dict(tasks))

        self.weighting_mode = self.config.weighting_mode
        if self.weighting_mode == "uncertainty":
            for name in self.tasks:
                self.register_parameter(
                    f"log_var_{name}", nn.Parameter(torch.zeros(()))
                )
        elif self.weighting_mode == "gradnorm":
            for name in self.tasks:
                self.register_parameter(
                    f"task_weight_{name}", nn.Parameter(torch.tensor(1.0))
                )

    # -- weighting ----------------------------------------------------------- #

    def task_weight_dict(self) -> dict[str, float]:
        """Current per-task weights (for logging / checkpointing)."""
        if self.weighting_mode == "fixed":
            return {
                "crop": self.config.crop_weight,
                "yield": self.config.yield_weight,
            }
        if self.weighting_mode == "uncertainty":
            return {
                name: float(torch.exp(-getattr(self, f"log_var_{name}")).item())
                for name in self.tasks
            }
        # gradnorm
        return {
            name: float(getattr(self, f"task_weight_{name}").item())
            for name in self.tasks
        }

    # -- forward ------------------------------------------------------------- #

    def per_task_losses(
        self,
        inputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute each task's raw loss on the current graph (no weighting)."""
        per_task: dict[str, torch.Tensor] = {}
        for name, criterion in self.tasks.items():
            if name not in inputs or name not in targets:
                raise LossBuildError(
                    f"task {name!r} requires both an input and a target"
                )
            per_task[name] = criterion(inputs[name], targets[name])
        return per_task

    def combine(
        self, per_task: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Weight the per-task losses into the total loss (current weights)."""
        if self.weighting_mode == "fixed":
            weights = {"crop": self.config.crop_weight,
                       "yield": self.config.yield_weight}
            total = torch.stack(
                [weights.get(name, 1.0) * value for name, value in per_task.items()]
            ).sum()
        elif self.weighting_mode == "uncertainty":
            terms = []
            for name, value in per_task.items():
                log_var = getattr(self, f"log_var_{name}")
                inv_var = torch.exp(-log_var)
                terms.append(0.5 * inv_var * value + 0.5 * log_var)
            total = torch.stack(terms).sum()
        else:  # gradnorm — weights are trainable parameters
            total = torch.stack(
                [getattr(self, f"task_weight_{name}") * value
                 for name, value in per_task.items()]
            ).sum()
        return total, dict(per_task)

    def forward(  # type: ignore[override]
        self,
        inputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted multi-task loss.

        Returns:
            ``(total_loss, per_task_losses)``.
        """
        return self.combine(self.per_task_losses(inputs, targets))


# --------------------------------------------------------------------------- #
# GradNorm (Chen et al., 2018)
# --------------------------------------------------------------------------- #


class GradNormController:
    """Optional GradNorm task-weight adaptation.

    GradNorm balances multi-task training by equalising the gradient norms of
    a shared encoder across tasks. After the trainer computes the per-task
    losses, :meth:`apply` updates the task weights inside a
    :class:`MultiTaskLoss` in ``gradnorm`` mode. The main backward pass then
    uses those updated weights.

    The shared-parameter scope is the model's ``shared_encoder`` (both task
    heads consume its representation). GradNorm loss only flows through the
    task-weight parameters — model gradients stay untouched here and are
    produced by the trainer's main backward pass.

    Args:
        model: The :class:`~ai.models.cropfusion.CropFusionModel`.
        multitask_loss: The :class:`MultiTaskLoss` in ``gradnorm`` mode.
        alpha: Asymmetry parameter (``0`` = uniform targets, ``1.5`` is the
            paper default).
        weight_lr: Learning rate for the task-weight parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        multitask_loss: MultiTaskLoss,
        *,
        alpha: float = 1.5,
        weight_lr: float = 0.01,
    ) -> None:
        self.model = model
        self.loss_module = multitask_loss
        self.alpha = float(alpha)
        self.weight_lr = float(weight_lr)
        self._initial_losses: dict[str, float] = {}
        self._weight_opt: torch.optim.Optimizer | None = None

    # -- internals ----------------------------------------------------------- #

    @property
    def task_names(self) -> list[str]:
        return list(self.loss_module.tasks.keys())

    def _shared_parameters(self) -> list[nn.Parameter]:
        shared = getattr(self.model, "shared_encoder", None)
        if shared is None:
            return [p for p in self.model.parameters() if p.requires_grad]
        return [p for p in shared.parameters() if p.requires_grad]

    def _get_weight_optimizer(self) -> torch.optim.Optimizer:
        if self._weight_opt is None:
            params = [
                getattr(self.loss_module, f"task_weight_{name}")
                for name in self.task_names
            ]
            self._weight_opt = torch.optim.Adam(params, lr=self.weight_lr)
        return self._weight_opt

    # -- public -------------------------------------------------------------- #

    def apply(
        self,
        per_task_losses: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        """Update the task weights from the current per-task losses.

        Args:
            per_task_losses: ``{task: scalar loss tensor}`` computed on the
                current batch's graph.

        Returns:
            The updated task weights (for logging).
        """
        names = self.task_names
        losses = {name: per_task_losses[name] for name in names}

        # Initial losses L_i(0) — recorded on the first call.
        if not self._initial_losses:
            with torch.no_grad():
                self._initial_losses = {
                    name: float(losses[name].detach().item()) for name in names
                }

        shared_params = self._shared_parameters()
        if not shared_params:
            raise LossBuildError("GradNorm requires trainable shared parameters")

        # Differentiable gradient norm of each task's weighted loss w.r.t. the
        # shared encoder. ``create_graph`` keeps G_i a function of w_i so the
        # GradNorm loss can update the weights (no ``.detach()`` here).
        grad_norms: dict[str, torch.Tensor] = {}
        for name in names:
            weight = getattr(self.loss_module, f"task_weight_{name}")
            scaled = weight * losses[name]
            grads = torch.autograd.grad(
                scaled,
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            squared = torch.stack(
                [g.norm(2) ** 2 for g in grads if g is not None]
            ).sum()
            grad_norms[name] = squared.sqrt()

        with torch.no_grad():
            current = {name: float(losses[name].detach().item()) for name in names}
            rates = {
                name: max(current[name], 1e-8) / max(self._initial_losses[name], 1e-8)
                for name in names
            }
            mean_rate = sum(rates.values()) / len(rates)
            relative = {name: rate / mean_rate for name, rate in rates.items()}

        # Targets G_i(t) * (r_i(t) ** alpha); the GradNorm loss keeps the
        # differentiable link to the task weights.
        targets = {
            name: grad_norms[name] * (relative[name] ** self.alpha) for name in names
        }
        gradnorm_loss = torch.stack(
            [torch.abs(grad_norms[name] - targets[name]) for name in names]
        ).sum()

        weight_params = [
            getattr(self.loss_module, f"task_weight_{name}") for name in names
        ]
        opt = self._get_weight_optimizer()
        opt.zero_grad()
        # Restrict backprop to the task weights (model grads untouched here).
        torch.autograd.backward(gradnorm_loss, inputs=weight_params, retain_graph=True)
        opt.step()

        # Renormalize so the weights sum to the number of tasks.
        with torch.no_grad():
            total = sum(getattr(self.loss_module, f"task_weight_{name}").item()
                        for name in names)
            if total > 0:
                scale = len(names) / total
                for name in names:
                    param = getattr(self.loss_module, f"task_weight_{name}")
                    param.mul_(scale)

        return self.loss_module.task_weight_dict()

    # -- state --------------------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        return {
            "initial_losses": dict(self._initial_losses),
            "weight_optimizer": (
                self._weight_opt.state_dict() if self._weight_opt else None
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._initial_losses = dict(state.get("initial_losses", {}))
        if state.get("weight_optimizer") is not None:
            self._weight_optimizer().load_state_dict(state["weight_optimizer"])


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def build_multi_task_loss(
    config: LossConfig | None = None,
    tasks: Mapping[str, nn.Module] | None = None,
) -> MultiTaskLoss:
    """Build a :class:`MultiTaskLoss` from a :class:`LossConfig`."""
    return MultiTaskLoss(config, tasks=tasks)
