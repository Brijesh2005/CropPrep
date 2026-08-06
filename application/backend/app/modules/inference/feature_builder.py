"""Feature Builder -> ModelInput (pipeline step 3-4).

Turns a :class:`ResolvedLocation` into the exact numeric feature vector the
exported model expects, using the ``feature_order`` declared in
``configs/model.yaml`` and the fitted ``scaler.pkl`` from the release
package. This mirrors the training-time feature contract without
re-implementing (or importing) the training feature-engineering code —
the feature order is a data contract (a list of names), not logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.core.exceptions import InferenceError
from app.modules.inference.location_resolver import ResolvedLocation
from inference_package.release import ReleasePackage


@dataclass(frozen=True, slots=True)
class ModelInput:
    tensor: torch.Tensor  # shape (1, num_features)
    feature_names: tuple[str, ...]
    raw_values: dict[str, float]


class FeatureBuilder:
    """Assembles and scales the feature vector for one prediction."""

    def __init__(self, package: ReleasePackage) -> None:
        self._package = package
        feature_order = package.model_config.get("feature_order")
        if not feature_order:
            raise InferenceError("configs/model.yaml is missing 'feature_order'")
        self._feature_order: tuple[str, ...] = tuple(feature_order)
        self._scaler = package.scaler
        self._device = package.device

    def build(self, location: ResolvedLocation) -> ModelInput:
        merged: dict[str, Any] = {
            **location.village_metadata,
            **location.historical_context,
            "lon": location.lon,
            "lat": location.lat,
        }

        raw_values: dict[str, float] = {}
        missing: list[str] = []
        for name in self._feature_order:
            value = merged.get(name)
            if value is None:
                missing.append(name)
                raw_values[name] = 0.0
            else:
                raw_values[name] = float(value)

        if missing:
            # Don't fail the request over a handful of missing optional fields —
            # the scaler was fit on real data including gaps — but surface it
            # for observability rather than silently guessing.
            from app.core.logging import get_logger

            get_logger("feature-builder").warning(
                "missing features filled with 0.0 before scaling",
                village=location.village,
                missing=missing,
            )

        vector = np.array([[raw_values[name] for name in self._feature_order]], dtype="float64")
        if self._scaler is not None and hasattr(self._scaler, "transform"):
            vector = self._scaler.transform(vector)

        tensor = torch.tensor(vector, dtype=torch.float32, device=self._device)
        return ModelInput(tensor=tensor, feature_names=self._feature_order, raw_values=raw_values)


__all__ = ["FeatureBuilder", "ModelInput"]
