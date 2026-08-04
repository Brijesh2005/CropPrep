"""Counterfactual ("what-if") explainer.

Answers questions like *"if rainfall increased, would the recommendation
change?"* by perturbing the sample and re-running the model:

* tabular perturbations — ``add`` / ``multiply`` / ``set`` a feature,
* image perturbations — scale the NDVI or EVI patches,
* temporal perturbations — mask (drop) an observation date.

Each result compares the original prediction against the perturbed one and
flags whether the crop recommendation changed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import CounterfactualConfig
from .exceptions import CounterfactualError
from .utils import single_sample_batch, to_numpy


class CounterfactualEngine:
    """Counterfactual perturbation engine for CropFusion."""

    def __init__(
        self,
        model: nn.Module,
        config: CounterfactualConfig | None = None,
        device: torch.device | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.config = config or CounterfactualConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.feature_names = list(feature_names or [])

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, sample: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Original / perturbed prediction for a sample."""
        batch = single_sample_batch(sample, self.device)
        out = self.model(batch)
        result: dict[str, Any] = {}
        if out.crop_logits is not None:
            probs = to_numpy(torch.softmax(out.crop_logits.float(), dim=-1))[0]
            result["crop_class"] = int(probs.argmax())
            result["crop_probs"] = probs
            result["crop_confidence"] = float(probs.max())
        if out.yield_pred is not None:
            result["yield_pred"] = float(out.yield_pred[0, 0].item())
        return result

    # ------------------------------------------------------------------ #
    # Perturbations
    # ------------------------------------------------------------------ #

    def _copy(self, sample: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}

    def perturb_tabular(
        self,
        sample: Mapping[str, torch.Tensor],
        feature: str | int,
        delta: float,
        mode: str = "add",
    ) -> dict[str, torch.Tensor]:
        """Perturb one tabular feature (by name or index)."""
        tabular = sample.get("tabular")
        if tabular is None:
            raise CounterfactualError("sample has no tabular tensor")
        out = self._copy(sample)
        vector = tabular.clone().float()
        index = self._resolve_feature(feature)
        if index >= vector.shape[0]:
            raise CounterfactualError(
                f"feature index {index} out of range ({vector.shape[0]})"
            )
        value = float(vector[index].item())
        if mode == "add":
            vector[index] = value + float(delta)
        elif mode == "multiply":
            vector[index] = value * float(delta)
        elif mode == "set":
            vector[index] = float(delta)
        else:
            raise CounterfactualError(f"unknown perturbation mode {mode!r}")
        out["tabular"] = vector
        return out

    def perturb_image(
        self,
        sample: Mapping[str, torch.Tensor],
        index: str = "ndvi",
        factor: float = 0.8,
        timestep: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Scale the NDVI / EVI patches (``factor < 1`` = vegetation decline)."""
        tensor = sample.get(index)
        if tensor is None:
            raise CounterfactualError(f"sample has no {index} tensor")
        out = self._copy(sample)
        modified = tensor.clone().float()
        if timestep is not None:
            modified[timestep] = modified[timestep] * float(factor)
        else:
            modified = modified * float(factor)
        out[index] = modified
        return out

    def mask_observation(
        self,
        sample: Mapping[str, torch.Tensor],
        timestep: int,
    ) -> dict[str, torch.Tensor]:
        """Mask one temporal observation (treated as padding)."""
        out = self._copy(sample)
        for key in ("ndvi", "evi"):
            tensor = sample.get(key)
            if tensor is not None and tensor.dim() >= 4 and timestep < tensor.shape[0]:
                out[key] = _zero_timestep(tensor, timestep)
        mask = out.get("temporal_mask")
        if mask is not None and timestep < mask.shape[0]:
            out["temporal_mask"] = mask.clone()
            out["temporal_mask"][timestep] = 0.0
        return out

    # ------------------------------------------------------------------ #
    # Explain
    # ------------------------------------------------------------------ #

    def explain(
        self,
        sample: Mapping[str, torch.Tensor],
        perturbations: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run a set of counterfactuals and compare against the original.

        Args:
            sample: The observation sample to perturb.
            perturbations: ``[{"feature"/"image"/"mask": ..., "delta": ...,
                "mode": ..., "label": ...}]``. Defaults to ``config.perturbations``.

        Returns:
            ``{"original": {...}, "counterfactuals": [...]}``.
        """
        specs = list(perturbations) if perturbations else self._default_perturbations(sample)
        specs = specs[: self.config.max_examples]
        original = self.predict(sample)
        results = []
        for spec in specs:
            label = str(spec.get("label", spec.get("feature", spec.get("index", "perturb"))))
            try:
                perturbed = self._apply(sample, spec)
                predicted = self.predict(perturbed)
            except CounterfactualError as exc:
                results.append({"label": label, "error": str(exc)})
                continue
            changed = self._crop_changed(original, predicted)
            results.append(
                {
                    "label": label,
                    "spec": dict(spec),
                    "crop_changed": changed,
                    "original_crop": original.get("crop_class"),
                    "new_crop": predicted.get("crop_class"),
                    "original_yield": original.get("yield_pred"),
                    "new_yield": predicted.get("yield_pred"),
                    "yield_delta": (
                        float(predicted.get("yield_pred", 0.0) - original.get("yield_pred", 0.0))
                        if predicted.get("yield_pred") is not None and original.get("yield_pred") is not None
                        else None
                    ),
                }
            )
        return {"original": original, "counterfactuals": results}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _apply(
        self, sample: Mapping[str, torch.Tensor], spec: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        if "feature" in spec:
            return self.perturb_tabular(
                sample, spec["feature"], spec.get("delta", 0.0), spec.get("mode", "add")
            )
        if "image" in spec:
            return self.perturb_image(
                sample, spec["image"], spec.get("factor", 0.8), spec.get("timestep")
            )
        if "mask" in spec:
            return self.mask_observation(sample, int(spec["mask"]))
        raise CounterfactualError("perturbation spec needs 'feature', 'image' or 'mask'")

    def _resolve_feature(self, feature: str | int) -> int:
        if isinstance(feature, int):
            return feature
        if not self.feature_names:
            raise CounterfactualError("no feature names provided; use a feature index")
        if feature in self.feature_names:
            return self.feature_names.index(feature)
        raise CounterfactualError(f"unknown feature {feature!r}", detail=self.feature_names)

    def _crop_changed(self, original: dict, predicted: dict) -> bool:
        if original.get("crop_class") is None or predicted.get("crop_class") is None:
            return False
        if original["crop_class"] != predicted["crop_class"]:
            return True
        # Same class but the margin shrank below the switch threshold.
        orig = np.asarray(original.get("crop_probs", []))
        pred = np.asarray(predicted.get("crop_probs", []))
        if orig.size and pred.size and orig.size == pred.size:
            margin_orig = orig[original["crop_class"]] - orig.max()
            margin_pred = pred[original["crop_class"]] - pred.max()
            if margin_pred - margin_orig < -self.config.switch_threshold:
                return True
        return False

    def _default_perturbations(
        self, sample: Mapping[str, torch.Tensor]
    ) -> list[dict[str, Any]]:
        configured = self.config.perturbations
        if configured:
            return [
                {"feature": key, **value, "label": key}
                for key, value in configured.items()
            ]
        defaults: list[dict[str, Any]] = []
        if "tabular" in sample and self.feature_names:
            for name in self.feature_names[:3]:
                defaults.append({"feature": name, "delta": 0.1, "mode": "multiply", "label": f"{name} +10%"})
        if "ndvi" in sample:
            defaults.append({"image": "ndvi", "factor": 0.7, "label": "NDVI -30%"})
        if "evi" in sample:
            defaults.append({"image": "evi", "factor": 0.7, "label": "EVI -30%"})
        return defaults


def _zero_timestep(tensor: torch.Tensor, timestep: int) -> torch.Tensor:
    modified = tensor.clone()
    modified[timestep] = 0.0
    return modified
