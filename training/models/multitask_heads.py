"""Multi-task heads — crop recommendation + yield prediction (+ future heads).

``CropHead`` is a multi-class classifier (logits over crop classes) and
``YieldHead`` is a single-value regressor. :class:`MultiTaskHeads` is a
registry container: heads are added by name and every head consumes the same
shared representation, so future tasks (crop health, disease detection, water
requirement) plug in via :meth:`MultiTaskHeads.add_head` without changing the
architecture.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
from torch import nn

from .exceptions import ModelConfigurationError
from .utils import get_activation


class CropHead(nn.Module):
    """Crop recommendation head (softmax classification over crop classes).

    Args:
        in_dim: Width of the shared representation.
        num_classes: Number of crop classes.
        hidden_dim: Width of the hidden layer (``None`` = ``in_dim``).
        dropout: Dropout before the classifier.
        activation: Hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ModelConfigurationError(
                "crop head requires num_classes >= 1", detail=num_classes
            )
        hidden = hidden_dim if hidden_dim is not None else in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            get_activation(activation),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, num_classes)
        self.output_dim = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return ``[B, num_classes]`` logits (softmax applied at inference)."""
        return self.classifier(self.net(x))


class YieldHead(nn.Module):
    """Yield prediction head (single-value regression).

    Args:
        in_dim: Width of the shared representation.
        hidden_dim: Width of the hidden layer (``None`` = ``in_dim``).
        dropout: Dropout before the output.
        activation: Hidden activation.
        output_clamp_min: Lower clamp on the prediction (``None`` = no clamp).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        output_clamp_min: float | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.output_clamp_min = output_clamp_min
        self.output_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return ``[B, 1]`` predicted yield."""
        out = self.net(x)
        if self.output_clamp_min is not None:
            out = torch.clamp(out, min=float(self.output_clamp_min))
        return out


class MultiTaskHeads(nn.Module):
    """Ordered registry of task heads sharing one representation.

    Args:
        heads: Optional mapping of ``name -> Head`` to start with.
    """

    def __init__(self, heads: Mapping[str, nn.Module] | None = None) -> None:
        super().__init__()
        self._heads = nn.ModuleDict(OrderedDict(heads or {}))

    # -- Registry API ------------------------------------------------------- #

    def add_head(self, name: str, module: nn.Module) -> "MultiTaskHeads":
        """Register a head under ``name`` (future tasks plug in here)."""
        if not isinstance(module, nn.Module):
            raise ModelConfigurationError(
                "head must be an nn.Module", detail=type(module).__name__
            )
        if name in self._heads:
            raise ModelConfigurationError(f"head {name!r} already registered")
        self._heads[name] = module
        return self

    def remove_head(self, name: str) -> "MultiTaskHeads":
        if name in self._heads:
            del self._heads[name]
        return self

    @property
    def names(self) -> list[str]:
        return list(self._heads.keys())

    def __getitem__(self, name: str) -> nn.Module:
        return self._heads[name]

    def __contains__(self, name: object) -> bool:
        return name in self._heads

    @property
    def output_dims(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        for name, head in self._heads.items():
            dims[name] = int(getattr(head, "output_dim", 0))
        return dims

    # -- nn.Module ---------------------------------------------------------- #

    def forward(self, shared: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """Run every head on the shared representation.

        Args:
            shared: ``[B, shared_dim]`` shared multimodal representation.

        Returns:
            ``{name: head_output}`` — ``crop_logits`` ``[B, C]`` and/or
            ``yield_pred`` ``[B, 1]``.
        """
        return {name: head(shared) for name, head in self._heads.items()}

    def named_head_outputs(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        """Alias head outputs to user-facing names (crop_logits / yield_pred)."""
        aliases: dict[str, Any] = {}
        for name, value in outputs.items():
            if name == "crop":
                aliases["crop_logits"] = value
            elif name == "yield":
                aliases["yield_pred"] = value
            else:
                aliases[f"{name}_output"] = value
        return aliases
