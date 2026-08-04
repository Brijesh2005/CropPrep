"""Integrated gradients (Sundararajan et al., 2017).

Attributions are computed along the path from a baseline to the input:

.. math::

    IG_i = (x_i - x'_i) \\int_0^1 \\frac{\\partial f(x' + \\alpha(x - x'))}
    {\\partial x_i} d\\alpha

Implemented for the tabular branch, the image (NDVI / EVI) patches, and the
shared multimodal embedding.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from .config import IntegratedGradientsConfig
from .exceptions import AttributionError
from .interfaces import AttributionMethod
from .utils import single_sample_batch, to_numpy


class IntegratedGradients(AttributionMethod):
    """Generic integrated-gradients over a differentiable forward path."""

    name = "integrated_gradients"

    def __init__(self, steps: int = 50) -> None:
        if steps < 2:
            raise AttributionError("integrated gradients requires steps >= 2")
        self.steps = int(steps)

    def attribute(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Attribute ``inputs`` w.r.t. ``target``.

        ``target`` must be a zero-dim or single-element tensor produced by a
        forward that depends on ``inputs`` (``inputs.requires_grad`` is set
        internally).
        """
        inputs = inputs.float().detach()
        baseline = baseline.float().detach().to(inputs.device)
        if inputs.shape != baseline.shape:
            raise AttributionError(
                "inputs and baseline shapes must match",
                detail=(tuple(inputs.shape), tuple(baseline.shape)),
            )
        delta = inputs - baseline
        gradients: list[torch.Tensor] = []
        for step in range(self.steps):
            alpha = step / (self.steps - 1)
            path = (baseline + alpha * delta).requires_grad_(True)
            self.model.zero_grad()
            scalar = self._forward(path)
            if not torch.is_tensor(scalar) or scalar.numel() != 1:
                raise AttributionError(
                    "forward must return a single scalar", detail=type(scalar)
                )
            scalar.backward()
            if path.grad is None:
                raise AttributionError("no gradient flowed to the input")
            gradients.append(path.grad.detach())
        avg_grad = torch.stack(gradients).mean(dim=0)
        return delta * avg_grad

    # Subclasses provide the forward path.
    def _forward(self, path: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _ModelIG(IntegratedGradients):
    """IG over the CropFusion model.

    Args:
        model: The trained model.
        forward_fn: ``callable(path_tensor) -> scalar`` building a batch from
            ``path``, running the model, and returning the target output.
    """

    def __init__(self, model: nn.Module, forward_fn: Callable[[torch.Tensor], torch.Tensor], steps: int = 50) -> None:
        super().__init__(steps=steps)
        self.model = model
        self._forward_fn = forward_fn

    def _forward(self, path: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(path)


def _target_scalar(out: Any, kind: str, class_index: int | None = None) -> torch.Tensor:
    if kind == "crop":
        logits = out.crop_logits
        if logits is None:
            raise AttributionError("model has no crop head")
        cls = class_index if class_index is not None else int(logits[0].argmax().item())
        return logits[0, cls]
    pred = out.yield_pred
    if pred is None:
        raise AttributionError("model has no yield head")
    return pred[0, 0]


def _baseline_for(
    config: IntegratedGradientsConfig,
    sample: Mapping[str, torch.Tensor],
    key: str,
    seed: int = 42,
) -> torch.Tensor:
    """Build a baseline tensor for ``key`` (tabular / ndvi / evi)."""
    value = sample[key].float()
    mode = config.baseline
    if mode == "zero":
        return torch.zeros_like(value)
    if mode == "mean":
        return torch.full_like(value, float(value.mean().item()))
    rng = np.random.RandomState(seed)
    frac = config.random_fraction
    noise = torch.from_numpy(rng.uniform(-frac, frac, size=value.shape).astype("float32"))
    return value * noise


def _make_forward(model: nn.Module, sample: Mapping[str, torch.Tensor], key: str, kind: str, class_index: int | None, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward closure replacing ``key`` in the sample batch."""

    def forward_fn(path: torch.Tensor) -> torch.Tensor:
        batch = single_sample_batch(sample, device)
        batch[key] = path.unsqueeze(0).to(device)
        out = model(batch)
        return _target_scalar(out, kind, class_index)

    return forward_fn


class TabularIntegratedGradients:
    """Integrated gradients for the tabular feature vector."""

    def __init__(
        self,
        model: nn.Module,
        config: IntegratedGradientsConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or IntegratedGradientsConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attribute(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        kind: str = "crop",
        class_index: int | None = None,
    ) -> np.ndarray:
        """Return ``[F]`` attributions for the tabular branch."""
        forward = _make_forward(self.model, sample, "tabular", kind, class_index, self.device)
        ig = _ModelIG(self.model, forward, steps=self.config.steps)
        baseline = _baseline_for(self.config, sample, "tabular")
        result = ig.attribute(sample["tabular"], baseline, torch.tensor(0.0))
        return to_numpy(result)


class ImageIntegratedGradients:
    """Integrated gradients for a single NDVI / EVI timestep patch."""

    def __init__(
        self,
        model: nn.Module,
        config: IntegratedGradientsConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or IntegratedGradientsConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attribute(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        index: str = "ndvi",
        timestep: int = 0,
        kind: str = "crop",
        class_index: int | None = None,
    ) -> np.ndarray:
        """Return ``[1, H, W]`` attributions for one patch."""
        patch = sample[index][timestep : timestep + 1]  # [1,1,H,W]
        forward = _make_forward(self.model, sample, index, kind, class_index, self.device)
        ig = _ModelIG(self.model, forward, steps=self.config.steps)
        baseline = _baseline_for(self.config, sample, index)
        baseline_patch = baseline[timestep : timestep + 1]
        result = ig.attribute(patch, baseline_patch, torch.tensor(0.0))
        return to_numpy(result)[0, 0]


class SharedEmbeddingIntegratedGradients:
    """Integrated gradients over the shared multimodal embedding.

    Attributes the ``[D]`` shared representation: starting from a zero
    embedding, how much does each latent dimension push toward the target?
    """

    def __init__(
        self,
        model: nn.Module,
        config: IntegratedGradientsConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or IntegratedGradientsConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attribute(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        kind: str = "crop",
        class_index: int | None = None,
    ) -> np.ndarray:
        """Return ``[D]`` attributions on the shared representation."""
        with torch.no_grad():
            out = self.model(single_sample_batch(sample, self.device))
            shared = out.shared_representation[0].float().detach()  # [D]

        def forward_fn(path: torch.Tensor) -> torch.Tensor:
            head_out = self.model.heads(path.unsqueeze(0))
            if kind == "crop":
                logits = head_out.get("crop")
                cls = class_index if class_index is not None else int(logits[0].argmax().item())
                return logits[0, cls]
            return head_out.get("yield")[0, 0]

        ig = _ModelIG(self.model, forward_fn, steps=self.config.steps)
        result = ig.attribute(shared, torch.zeros_like(shared), torch.tensor(0.0))
        return to_numpy(result)
