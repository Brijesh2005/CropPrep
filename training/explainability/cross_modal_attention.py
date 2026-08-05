"""Cross-modal attention explainer.

Explains the interaction between the image (temporal) and tabular branches:

* the cross-attention score (image embedding attends to the tabular embedding),
* the per-modality **gates** from the adaptive gated fusion — which modality
  contributed more,
* tabular feature-token importance (TabTransformer CLS attention),
* temporal observation importance (temporal-transformer attention),
* a cross-modal contribution heatmap ``[timesteps x tabular features]``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .config import CrossModalConfig
from .exceptions import AttentionError
from .utils import AttentionCapture, single_sample_batch, to_numpy


class CrossModalExplainer:
    """Cross-modal attention + modality-gate explainer."""

    def __init__(
        self,
        model: nn.Module,
        config: CrossModalConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or CrossModalConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.capture = AttentionCapture()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rollout(layer_weights: list[torch.Tensor], include_residual: bool = True) -> torch.Tensor:
        if not layer_weights:
            raise AttentionError("rollout requires at least one layer")
        n = layer_weights[0].shape[-1]
        identity = torch.eye(n, device=layer_weights[0].device)
        result: torch.Tensor | None = None
        for weights in layer_weights:
            matrix = 0.5 * weights + 0.5 * identity if include_residual else weights
            result = matrix if result is None else result @ matrix
        assert result is not None
        return result

    def _install(self) -> None:
        cross = getattr(getattr(self.model, "cross_attention", None), "attention", None)
        if cross is not None:
            self.capture.register(cross, "cross")
        tab = getattr(self.model, "tab_encoder", None)
        for i, block in enumerate(getattr(tab, "blocks", []) or []):
            self.capture.register(block.self_attn, f"tab_{i}")
        temporal = getattr(self.model, "temporal_transformer", None)
        for i, block in enumerate(getattr(temporal, "blocks", []) or []):
            self.capture.register(block.self_attn, f"temporal_{i}")

    # ------------------------------------------------------------------ #
    # Explain
    # ------------------------------------------------------------------ #

    def explain(
        self,
        sample: Mapping[str, torch.Tensor],
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Cross-modal explanation for one sample."""
        self._install()
        batch = single_sample_batch(sample, self.device)
        out = self.model(batch)

        # -- Per-modality gates ------------------------------------------ #
        gates: dict[str, float] = {}
        for key, value in (getattr(out, "gates", {}) or {}).items():
            if torch.is_tensor(value):
                gates[key] = float(value.reshape(-1)[0].item())

        # -- Cross-attention score --------------------------------------- #
        cross_w = self.capture.weights("cross", headwise=False)
        cross_score = float(cross_w[0, 0, 0].item()) if cross_w is not None else None

        # -- Tabular feature-token importance ---------------------------- #
        captured = self.capture.captured_names()
        tab_names = [n for n in captured if n.startswith("tab_")]
        tab_weights = [
            self.capture.weights(n, headwise=False) for n in tab_names
        ]
        feature_importance: np.ndarray | None = None
        tab_rollout: np.ndarray | None = None
        if tab_weights and all(w is not None for w in tab_weights):
            tab_rollout = to_numpy(self._rollout([w for w in tab_weights if w is not None]))
            cls_row = tab_rollout[0, 0]  # [1 + F]
            feature_importance = cls_row[1:]
            if self.config.normalize and feature_importance.size:
                s = float(feature_importance.sum())
                if s > 0:
                    feature_importance = feature_importance / s

        # -- Temporal observation importance ----------------------------- #
        temporal_names = [n for n in captured if n.startswith("temporal_")]
        temporal_weights = [
            self.capture.weights(n, headwise=False) for n in temporal_names
        ]
        obs_importance: np.ndarray | None = None
        temporal_rollout: np.ndarray | None = None
        if temporal_weights and all(w is not None for w in temporal_weights):
            temporal_rollout = to_numpy(self._rollout([w for w in temporal_weights if w is not None]))
            obs_importance = temporal_rollout[0, 0, 1:]
            mask = to_numpy(batch.get("temporal_mask"))[0]
            if mask.shape[0] == obs_importance.shape[0]:
                obs_importance = obs_importance * mask

        # -- Cross-modal contribution heatmap [T, F] --------------------- #
        heatmap: np.ndarray | None = None
        if obs_importance is not None and feature_importance is not None:
            heatmap = np.outer(obs_importance, feature_importance)
            if cross_score is not None:
                heatmap = heatmap * cross_score
            if self.config.normalize and heatmap.size and heatmap.max() > 0:
                heatmap = heatmap / float(heatmap.max())

        return {
            "cross_attention_score": cross_score,
            "gates": gates,
            "feature_importance": feature_importance,
            "observation_importance": obs_importance,
            "cross_modal_heatmap": heatmap,
            "tab_rollout": tab_rollout,
            "temporal_rollout": temporal_rollout,
            "feature_names": feature_names,
        }
