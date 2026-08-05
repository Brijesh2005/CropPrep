"""Ports / interfaces for the explainability package.

Keeps the explainers dependant on abstractions (ports) so CAM methods,
attribution methods and confidence estimators can be swapped without touching
the surrounding code (dependency inversion).

* :class:`CamMethod` — combines a feature-map activation and its gradient into
  a class-activation map (GradCAM / GradCAM++ / EigenCAM / LayerCAM).
* :class:`AttributionMethod` — attributes a prediction to its inputs
  (integrated gradients, SHAP).
* :class:`ConfidenceEstimator` — produces a calibrated confidence for a
  prediction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


class CamMethod(ABC):
    """Computes a class-activation map from activations and gradients.

    Args:
        relu: Apply ReLU to the raw CAM (keep only positive evidence).
    """

    name: str = "cam"

    def __init__(self, relu: bool = True) -> None:
        self.relu = relu

    @abstractmethod
    def weights(
        self, activations: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        """Return per-channel importance weights.

        Args:
            activations: ``[B, C, H, W]`` target-layer activations.
            gradients: ``[B, C, H, W]`` gradient of the target output w.r.t.
                the activations.

        Returns:
            ``[B, C]`` channel weights.
        """

    def compute(
        self, activations: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        """Build the spatial CAM ``[B, H, W]`` for a batch."""
        weights = self.weights(activations, gradients)
        cam = torch.einsum("bc,bchw->bhw", weights, activations)
        if self.relu:
            cam = torch.relu(cam)
        return cam


class AttributionMethod(ABC):
    """Attributes a prediction to its inputs."""

    name: str = "attribution"

    @abstractmethod
    def attribute(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return an attribution with the same shape as ``inputs``.

        Args:
            inputs: The input tensor to explain.
            baseline: The reference input (e.g. zeros or a background sample).
            target: The target output (e.g. predicted class index).
        """


class ConfidenceEstimator(ABC):
    """Estimates the confidence of a model prediction."""

    name: str = "confidence"

    @abstractmethod
    def estimate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a dict of confidence / uncertainty metrics."""
