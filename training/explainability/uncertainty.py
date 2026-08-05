"""Prediction confidence and uncertainty estimation.

* Prediction confidence (top-1 softmax probability) and predictive entropy for
  the crop task.
* Monte-Carlo dropout (optional) for both tasks — runs the model several times
  with dropout active to get a prediction distribution.
* Expected Calibration Error (ECE) with a reliability diagram.
* Confidence distribution over a set of samples.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import UncertaintyConfig
from .exceptions import UncertaintyError
from .utils import compute_probability, outputs_to_task, to_numpy


def _is_dropout(module: nn.Module) -> bool:
    return isinstance(module, nn.Dropout)


class _McDropoutContext:
    """Temporarily enable Dropout (keeping BatchNorm in eval) for MC sampling."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._dropouts: list[nn.Dropout] = []
        self._states: list[bool] = []

    def __enter__(self) -> "_McDropoutContext":
        self.model.eval()  # BatchNorm stays eval (single-sample safe)
        for module in self.model.modules():
            if _is_dropout(module):
                self._dropouts.append(module)
                self._states.append(module.training)
                module.training = True
        return self

    def __exit__(self, *exc: Any) -> None:
        for module, state in zip(self._dropouts, self._states):
            module.training = state


class UncertaintyEstimator:
    """Estimates confidence / uncertainty for CropFusion predictions.

    Args:
        model: The trained :class:`~ai.models.cropfusion.CropFusionModel`.
        config: Validated :class:`UncertaintyConfig`.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: UncertaintyConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or UncertaintyConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------ #
    # Crop confidence
    # ------------------------------------------------------------------ #

    def _probs1d(self, logits: torch.Tensor) -> np.ndarray:
        probs = compute_probability(logits)
        return probs[0] if probs.ndim == 2 else probs

    def crop_confidence(self, logits: torch.Tensor) -> float:
        """Top-1 softmax probability for the predicted class."""
        return float(self._probs1d(logits).max())

    def entropy(self, logits: torch.Tensor) -> float:
        """Predictive entropy (nats) of the crop distribution."""
        probs = self._probs1d(logits)
        with np.errstate(divide="ignore"):
            return float(-np.sum(probs * np.log(probs)))

    # ------------------------------------------------------------------ #
    # Yield confidence
    # ------------------------------------------------------------------ #

    def yield_confidence(
        self, prediction: float, uncertainty: float, scale: float = 1.0
    ) -> float:
        """Confidence = ``1 - normalized uncertainty`` clipped to ``[0, 1]``.

        ``scale`` is the target std (or a reference residual) used to
        normalise the uncertainty.
        """
        if scale <= 0 or uncertainty < 0:
            return 1.0
        return float(max(0.0, min(1.0, 1.0 - uncertainty / scale)))

    # ------------------------------------------------------------------ #
    # MC dropout
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def mc_dropout(
        self,
        batch: Mapping[str, torch.Tensor],
        samples: int | None = None,
    ) -> dict[str, Any]:
        """Run ``samples`` forward passes with dropout active.

        Args:
            batch: A ``batch_size=1`` model batch.
            samples: Number of passes (``None`` = config value).

        Returns:
            ``crop_probs`` ``[samples, C]``, ``yield_preds`` ``[samples]`` and
            the means / standard deviations.
        """
        samples = samples if samples is not None else self.config.mc_dropout_samples
        if samples < 1:
            raise UncertaintyError(
                "mc_dropout requires mc_dropout_samples >= 1", detail=samples
            )
        model = self.model
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        crop_logits: list[torch.Tensor] = []
        yield_preds: list[torch.Tensor] = []
        with _McDropoutContext(model):
            for _ in range(samples):
                out = model(batch)
                tasks = outputs_to_task(out)
                if "crop" in tasks:
                    crop_logits.append(tasks["crop"].float())
                if "yield" in tasks:
                    yield_preds.append(tasks["yield"].float())

        result: dict[str, Any] = {}
        if crop_logits:
            logits = torch.stack(crop_logits)  # [S, 1, C]
            probs = torch.softmax(logits, dim=-1).mean(dim=0)  # [1, C]
            result["crop_probs"] = to_numpy(probs)[0]
            result["crop_pred"] = int(probs.argmax(-1).item())
            result["crop_conf"] = float(probs.max(-1).values.item())
            result["crop_entropy"] = float(
                -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=-1).item()
            )
        if yield_preds:
            preds = torch.stack(yield_preds).flatten()  # [S]
            result["yield_mean"] = float(preds.mean().item())
            result["yield_std"] = float(preds.std().item())
        return result

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def calibration(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        bins: int | None = None,
    ) -> dict[str, Any]:
        """Expected calibration error + reliability bins.

        Args:
            confidences: Per-sample top-1 confidence in ``[0, 1]``.
            correct: Per-sample boolean correctness.
            bins: Number of equal-width bins (``None`` = config value).

        Returns:
            ``{"ece": float, "bins": {bin_center: {"confidence", "accuracy",
            "count"}}}``.
        """
        bins = bins or self.config.bins
        confidences = np.asarray(confidences, dtype="float64")
        correct = np.asarray(correct, dtype="float64")
        if confidences.size == 0:
            raise UncertaintyError("cannot calibrate on empty predictions")

        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_data: dict[str, dict[str, float]] = {}
        ece = 0.0
        for i in range(bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == bins - 1:
                mask |= confidences == 1.0
            count = int(mask.sum())
            if count == 0:
                continue
            bin_conf = float(confidences[mask].mean())
            bin_acc = float(correct[mask].mean())
            center = float((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_data[str(round(center, 3))] = {
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "count": count,
            }
            ece += (count / confidences.size) * abs(bin_acc - bin_conf)
        return {"ece": float(ece), "bins": bin_data}

    def confidence_distribution(
        self, confidences: np.ndarray, bins: int | None = None
    ) -> dict[str, Any]:
        """Histogram of confidence values (for the confidence plot)."""
        bins = bins or self.config.bins
        counts, edges = np.histogram(
            np.asarray(confidences, dtype="float64"), bins=bins, range=(0.0, 1.0)
        )
        return {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
        }
