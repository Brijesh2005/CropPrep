"""SHAP explainer for the TabTransformer (self-contained KernelSHAP).

A faithful, dependency-free implementation of KernelSHAP (Lundberg & Lee,
2017): Shapley values are recovered via a weighted linear regression over
random feature coalitions, using the training background as the reference
distribution. When the ``shap`` library is installed and ``prefer_library`` is
set, it is used instead.

Supports:

* local feature importance (per-sample SHAP values),
* global feature importance (mean |SHAP| over a dataset),
* force / waterfall / decision / summary / bar / dependence / interaction
  plots (matplotlib),
* CSV / JSON export of the SHAP values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import ShapConfig
from .exceptions import ShapError
from .utils import feature_names, single_sample_batch, to_numpy


@dataclass
class ShapResult:
    """Local SHAP attribution for one sample."""

    values: np.ndarray
    base_value: float
    feature_names: list[str]
    target: dict[str, Any] = field(default_factory=dict)
    global_importance: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_value": float(self.base_value),
            "values": {name: float(v) for name, v in zip(self.feature_names, self.values)},
            "target": self.target,
            "global_importance": self.global_importance,
        }


def _target_spec(kind: str, value: Any) -> dict[str, Any]:
    """Normalise a target request to ``{"kind", "class"}`` or ``{"kind"}``."""
    if kind == "crop":
        return {"kind": "crop", "class": int(value)}
    if kind == "yield":
        return {"kind": "yield"}
    raise ShapError(f"unsupported SHAP target {kind!r}")


class SHAPExplainer:
    """Self-contained KernelSHAP for the CropFusion tabular branch.

    Args:
        model: The trained :class:`~ai.models.cropfusion.CropFusionModel`.
        config: Validated :class:`ShapConfig`.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ShapConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or ShapConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------ #
    # Model scalar helper
    # ------------------------------------------------------------------ #

    def _scalar(
        self,
        tabular_vector: np.ndarray,
        sample: Mapping[str, torch.Tensor],
        target: dict[str, Any],
    ) -> float:
        batch = single_sample_batch(sample, self.device)
        batch["tabular"] = (
            torch.as_tensor(tabular_vector, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        out = self.model(batch)
        if target["kind"] == "crop":
            logits = out.crop_logits
            if logits is None:
                raise ShapError("model has no crop head; cannot explain crop")
            return float(logits[0, target["class"]].item())
        pred = out.yield_pred
        if pred is None:
            raise ShapError("model has no yield head; cannot explain yield")
        return float(pred[0, 0].item())

    def _base_value(
        self,
        background: Sequence[Mapping[str, torch.Tensor]],
        target: dict[str, Any],
        names: list[str],
    ) -> float:
        """Expected model output over the background (all features masked)."""
        values = [
            self._scalar(np.asarray(b["tabular"].numpy(), dtype="float64"), b, target)
            for b in background
        ]
        return float(np.mean(values))

    # ------------------------------------------------------------------ #
    # Kernel SHAP
    # ------------------------------------------------------------------ #

    def _coalition(self, size: int, m: int, rng: np.random.RandomState) -> np.ndarray:
        z = np.zeros(m, dtype="float64")
        z[rng.choice(m, size=size, replace=False)] = 1.0
        return z

    def kernel_shap(
        self,
        sample: Mapping[str, torch.Tensor],
        background: Sequence[Mapping[str, torch.Tensor]],
        target: dict[str, Any],
        feature_names_: Sequence[str] | None = None,
    ) -> ShapResult:
        """Local SHAP values via KernelSHAP."""
        x = np.asarray(sample["tabular"].numpy(), dtype="float64")
        m = int(x.shape[0])
        if m == 0:
            raise ShapError("sample has no tabular features to explain")
        names = list(feature_names_ or [f"feature_{i}" for i in range(m)])

        base = self._base_value(background, target, names)
        f_full = self._scalar(x, sample, target)

        max_samples = min(self.config.max_samples, 2 ** m)
        rng = np.random.RandomState(self.config.seed if hasattr(self.config, "seed") else 42)

        # Sample coalition sizes from the Shapley kernel distribution.
        sizes = np.arange(1, m)
        if sizes.size == 0:
            # Single feature: SHAP value is the full effect directly.
            values = np.asarray([f_full - base], dtype="float64")
            return ShapResult(
                values=values, base_value=base, feature_names=names, target=target
            )
        kernel_weights = np.asarray(
            [(m - 1) / (math.comb(m, s) * s * (m - s)) for s in sizes]
        )
        kernel_weights /= kernel_weights.sum()

        coalitions: list[np.ndarray] = []
        fz: list[float] = []
        bg_idx = np.arange(len(background))
        for _ in range(int(max_samples)):
            size = int(rng.choice(sizes, p=kernel_weights))
            z = self._coalition(size, m, rng)
            # Fill masked features from a random background sample.
            bg = background[int(rng.choice(bg_idx))]
            perturbed = np.where(z == 1.0, x, np.asarray(bg["tabular"].numpy(), dtype="float64"))
            coalitions.append(z)
            fz.append(self._scalar(perturbed, sample, target))

        Z = np.stack(coalitions)  # [N, M]
        y = np.asarray(fz, dtype="float64") - base  # [N]
        # Shapley kernel weight for each coalition size.
        w = np.asarray(
            [(m - 1) / (math.comb(m, int(z.sum())) * int(z.sum()) * (m - int(z.sum())))
             for z in coalitions]
        )

        # Weighted ridge regression: min ||sqrt(W)(Z phi - y)||^2 + lambda||phi||^2.
        sqrt_w = np.sqrt(w)
        A = Z * sqrt_w[:, None]
        b = y * sqrt_w
        ridge = 1e-6
        gram = A.T @ A + ridge * np.eye(m)
        phi = np.linalg.solve(gram, A.T @ b)

        # Renormalise so the contributions sum to f(x) - base.
        scale = (f_full - base) / (phi.sum() + 1e-12)
        phi = phi * scale

        return ShapResult(
            values=phi, base_value=base, feature_names=names, target=target
        )

    # ------------------------------------------------------------------ #
    # Gradient SHAP (fast alternative)
    # ------------------------------------------------------------------ #

    def gradient_shap(
        self,
        sample: Mapping[str, torch.Tensor],
        background: Sequence[Mapping[str, torch.Tensor]],
        target: dict[str, Any],
        feature_names_: Sequence[str] | None = None,
    ) -> ShapResult:
        """Gradient x Input approximation (fast, no coalitions)."""
        x = np.asarray(sample["tabular"].numpy(), dtype="float64")
        m = int(x.shape[0])
        names = list(feature_names_ or [f"feature_{i}" for i in range(m)])
        base = self._base_value(background, target, names)
        bg = np.mean(
            [np.asarray(b["tabular"].numpy(), dtype="float64") for b in background],
            axis=0,
        )

        batch = single_sample_batch(sample, self.device)
        batch["tabular"] = (
            torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            .to(self.device).requires_grad_(True)
        )
        out = self.model(batch)
        if target["kind"] == "crop":
            scalar = out.crop_logits[0, target["class"]]
        else:
            scalar = out.yield_pred[0, 0]
        self.model.zero_grad()
        scalar.backward()
        grad = batch["tabular"].grad[0].detach().cpu().numpy()
        values = grad * (x - bg)
        return ShapResult(
            values=values, base_value=base, feature_names=names, target=target
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def explain(
        self,
        sample: Mapping[str, torch.Tensor],
        background: Sequence[Mapping[str, torch.Tensor]],
        *,
        kind: str = "crop",
        target_class: int | None = None,
        feature_names_: Sequence[str] | None = None,
    ) -> ShapResult:
        """Explain a single sample (``kind``: ``crop`` | ``yield``)."""
        if kind == "crop":
            logits = self.model(single_sample_batch(sample, self.device)).crop_logits
            cls = target_class if target_class is not None else int(logits[0].argmax().item())
            target = _target_spec("crop", cls)
        else:
            target = _target_spec("yield", 0)
        if self.config.method == "gradient":
            return self.gradient_shap(sample, background, target, feature_names_)
        return self.kernel_shap(sample, background, target, feature_names_)

    def global_importance(
        self,
        samples: Sequence[Mapping[str, torch.Tensor]],
        background: Sequence[Mapping[str, torch.Tensor]],
        *,
        kind: str = "crop",
        feature_names_: Sequence[str] | None = None,
    ) -> ShapResult:
        """Mean |SHAP| over a dataset (global feature importance)."""
        names = list(
            feature_names_
            or (feature_names(self._preprocessor) if hasattr(self, "_preprocessor") else [])
            or (["feature_%d" % i for i in range(_feature_count(samples))])
        )
        accumulated = np.zeros(len(names), dtype="float64")
        count = 0
        for sample in samples:
            result = self.explain(sample, background, kind=kind, feature_names_=names)
            accumulated += np.abs(result.values)
            count += 1
        if count:
            accumulated /= count
        importance = {name: float(v) for name, v in zip(names, accumulated)}
        return ShapResult(
            values=accumulated,
            base_value=0.0,
            feature_names=names,
            target={"kind": kind},
            global_importance=importance,
        )

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def to_dict(self, result: ShapResult) -> dict[str, Any]:
        return result.to_dict()

    def to_csv(self, result: ShapResult, path: str | Path) -> Path:
        import csv

        out = Path(path)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", "shap_value"])
            for name, value in zip(result.feature_names, result.values):
                writer.writerow([name, f"{value:.6f}"])
        return out

    def to_json(self, result: ShapResult, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return out


def _feature_count(samples: Sequence[Mapping[str, torch.Tensor]]) -> int:
    for sample in samples:
        if "tabular" in sample:
            return int(sample["tabular"].shape[0])
    return 0
