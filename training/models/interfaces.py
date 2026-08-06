"""Ports / interfaces for the AI model package.

Keeps the multimodal architecture dependant on abstractions (ports) rather
than concrete implementations so encoders, fusion blocks and heads can be
swapped without touching the surrounding code (dependency inversion).

Implemented concretely in ``backbone.py``, ``tabtransformer.py``,
``multitask_heads.py`` and ``losses.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ImageEncoder(nn.Module, ABC):
    """Port for a per-timestep image-sequence encoder.

    Consumes an image sequence ``[B, T, C, H, W]`` (C == 1 for vegetation
    index patches) and returns per-timestep feature vectors ``[B, T, D]``.
    """

    #: Dimensionality of each timestep's feature vector.
    feature_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Encode a sequence of image patches.

        Args:
            x: ``[B, T, C, H, W]`` patch tensor.

        Returns:
            ``[B, T, feature_dim]`` per-timestep features.
        """


class Head(nn.Module, ABC):
    """Port for a task head operating on the shared representation."""

    #: Dimensionality of the head output (logits / regression scalar).
    output_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Map the shared representation to task outputs.

        Args:
            x: ``[B, shared_dim]`` shared multimodal representation.

        Returns:
            Head output (``[B, num_classes]`` or ``[B, 1]``).
        """


class TaskLoss(nn.Module, ABC):
    """Port for a per-task loss.

    Losses are pure function objects — they never update model parameters
    (that belongs to Phase 6 training). Each implementation just computes a
    scalar loss from inputs and targets.
    """

    @abstractmethod
    def forward(  # type: ignore[override]
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss for one task.

        Args:
            inputs: Model output for the task (``[B, C]`` logits or ``[B, 1]``).
            targets: Ground-truth for the task (``[B]`` int64 or ``[B]`` float).

        Returns:
            Scalar loss (0-dim tensor).
        """
