"""Loss interfaces for the multi-task model.

Implements the Phase 5 loss interfaces **without training** — each loss is a
pure function object computing a scalar from model outputs and targets. Model
parameters are never updated here (optimizers / schedulers belong to Phase 6).

Supported:

* :class:`CrossEntropyLoss` — cross entropy for crop classification.
* :class:`LabelSmoothingLoss` — cross entropy with soft label smoothing.
* :class:`FocalLoss` — focal loss for imbalanced crop classes (Lin et al. 2017).
* :class:`MSELoss` — mean squared error for yield regression.
* :class:`HuberLoss` — smooth L1 / Huber for robust yield regression.
* :class:`WeightedMultiTaskLoss` — combines per-task losses with fixed or
  learnable (Kendall-style uncertainty) weights.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import LossConfig
from .exceptions import ModelConfigurationError


def _align_regression(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squeeze ``[B, 1]`` regression outputs to ``[B]`` and force float32."""
    pred = inputs
    target = targets
    if pred.dim() == 2 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 2 and target.size(1) == 1:
        target = target.squeeze(1)
    return pred.float(), target.float()


class CrossEntropyLoss(nn.Module):
    """Cross entropy for crop-class logits.

    Args:
        weight: Optional ``[C]`` class weights for imbalanced crops.
        reduction: ``mean`` | ``sum``.
    """

    def __init__(self, weight: torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """``inputs`` ``[B, C]`` logits, ``targets`` ``[B]`` int64."""
        return self.criterion(inputs, targets)


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with soft label smoothing (self-contained implementation).

    Args:
        smoothing: Smoothing factor in ``[0, 1)``.
        reduction: ``mean`` | ``sum``.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ModelConfigurationError(
                "label smoothing must be in [0, 1)", detail=smoothing
            )
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, fill_value=self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1).long(), 1.0 - self.smoothing)

        loss = -(true_dist * log_probs).sum(dim=-1)
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) for imbalanced crops.

    Args:
        gamma: Focusing parameter (``2.0`` is the standard value).
        alpha: Optional ``[C]`` class weights.
        reduction: ``mean`` | ``sum``.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean") -> None:
        super().__init__()
        if gamma < 0:
            raise ModelConfigurationError("gamma must be >= 0", detail=gamma)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()
        target_probs = probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)
        loss = -(1.0 - target_probs) ** self.gamma * target_log_probs
        if self.alpha is not None:
            weights = self.alpha.to(loss.device).gather(0, targets.long())
            loss = loss * weights
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class MSELoss(nn.Module):
    """Mean squared error for yield regression."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred, target = _align_regression(inputs, targets)
        squared = (pred - target) ** 2
        if self.reduction == "sum":
            return squared.sum()
        return squared.mean()


class HuberLoss(nn.Module):
    """Smooth L1 / Huber loss for robust yield regression.

    Args:
        beta: Threshold at which the loss transitions from quadratic to
            linear (``1.0`` = default Huber, ``0.5`` = SmoothL1 default).
        reduction: ``mean`` | ``sum``.
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.criterion = nn.SmoothL1Loss(beta=beta, reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred, target = _align_regression(inputs, targets)
        return self.criterion(pred, target)


class WeightedMultiTaskLoss(nn.Module):
    """Combines per-task losses with fixed or learnable weights.

    ``weighting_mode="fixed"`` uses the configured task weights;
    ``weighting_mode="learnable"`` uses Kendall et al. (2018) Gaussian
    uncertainty weighting, where each task's inverse variance is learned
    alongside the model.

    Args:
        config: Validated :class:`LossConfig` (defines task losses + weights).
        tasks: Mapping of task name -> concrete :class:`TaskLoss`. Built from
            ``config`` when omitted.

    Examples::

        tasks = {"crop": LabelSmoothingLoss(), "yield": HuberLoss()}
        loss = WeightedMultiTaskLoss(config, tasks)
        total, per_task = loss(
            {"crop": logits, "yield": yield_pred},
            {"crop": crop_labels, "yield": yield_labels},
        )
    """

    def __init__(
        self,
        config: LossConfig | None = None,
        tasks: Mapping[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LossConfig()
        if tasks is None:
            tasks = self._default_tasks()
        if not tasks:
            raise ModelConfigurationError("WeightedMultiTaskLoss requires >= 1 task")
        self.tasks = nn.ModuleDict(dict(tasks))

        self.weighting_mode = self.config.weighting_mode
        if self.weighting_mode == "learnable":
            for name in self.tasks:
                self.register_parameter(
                    f"log_var_{name}",
                    nn.Parameter(torch.zeros(())),
                )

    def _default_tasks(self) -> dict[str, nn.Module]:
        cfg = self.config
        tasks: dict[str, nn.Module] = {}
        tasks["crop"] = {
            "cross_entropy": lambda: CrossEntropyLoss(reduction=cfg.reduction),
            "label_smoothing": lambda: LabelSmoothingLoss(
                smoothing=cfg.label_smoothing, reduction=cfg.reduction
            ),
            "focal": lambda: FocalLoss(gamma=cfg.focal_gamma, reduction=cfg.reduction),
        }[cfg.crop_loss]()
        tasks["yield"] = {
            "mse": lambda: MSELoss(reduction=cfg.reduction),
            "huber": lambda: HuberLoss(reduction=cfg.reduction),
        }[cfg.yield_loss]()
        return tasks

    # -- nn.Module ---------------------------------------------------------- #

    def forward(  # type: ignore[override]
        self,
        inputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted multi-task loss.

        Args:
            inputs: Mapping of task name -> model output.
            targets: Mapping of task name -> ground truth.

        Returns:
            ``(total_loss, per_task_losses)``.
        """
        per_task: dict[str, torch.Tensor] = {}
        for name, criterion in self.tasks.items():
            if name not in inputs or name not in targets:
                raise ModelConfigurationError(
                    f"task {name!r} requires both an input and a target "
                    "(multi-task loss)"
                )
            per_task[name] = criterion(inputs[name], targets[name])

        if self.weighting_mode == "fixed":
            weights = {
                "crop": self.config.crop_weight,
                "yield": self.config.yield_weight,
            }
            total = torch.stack(
                [weights.get(name, 1.0) * value for name, value in per_task.items()]
            ).sum()
        else:
            # Kendall-style learnable weighting (Gaussian uncertainty).
            terms = []
            for name, value in per_task.items():
                log_var = getattr(self, f"log_var_{name}")
                inv_var = torch.exp(-log_var)
                terms.append(0.5 * inv_var * value + 0.5 * log_var)
            total = torch.stack(terms).sum()
        return total, per_task
