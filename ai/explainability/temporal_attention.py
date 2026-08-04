"""Temporal attention explainer for the temporal transformer.

Uses attention rollout (Abnar & Zuidema, 2020) over the temporal transformer's
self-attention layers to answer *which observation dates contributed most* to
the image embedding, and thereby to the prediction.

* attention maps — per-layer (and optionally per-head) attention weights,
* attention rollout — the residual-aware product across layers,
* temporal importance — the CLS attention to each timestep,
* observation ranking — timesteps ordered by importance.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .config import TemporalAttentionConfig
from .exceptions import AttentionError
from .utils import AttentionCapture, single_sample_batch, to_numpy


class TemporalAttentionExplainer:
    """Temporal importance from the ``temporal_transformer`` attention."""

    def __init__(
        self,
        model: nn.Module,
        config: TemporalAttentionConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or TemporalAttentionConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not getattr(model, "use_image", False):
            raise AttentionError("TemporalAttentionExplainer requires an image branch")
        self.capture = AttentionCapture()

    # ------------------------------------------------------------------ #
    # Attention extraction
    # ------------------------------------------------------------------ #

    def _install(self) -> list[str]:
        tt = getattr(self.model, "temporal_transformer", None)
        if tt is None or not getattr(tt, "blocks", None):
            raise AttentionError("model has no temporal transformer blocks")
        names: list[str] = []
        for i, block in enumerate(tt.blocks):
            name = f"temporal_{i}"
            self.capture.register(block.self_attn, name)
            names.append(name)
        return names

    def _layer_weights(self) -> list[torch.Tensor]:
        names = self.capture.captured_names()
        if not names:
            raise AttentionError("no temporal attention weights captured")
        weights = []
        for name in names:
            w = self.capture.weights(name, headwise=False)
            if w is None:
                raise AttentionError(f"no weights for {name}")
            weights.append(w)  # [B, T', T']
        return weights

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #

    def rollout(self, layer_weights: list[torch.Tensor]) -> torch.Tensor:
        """Residual-aware attention rollout across layers (Abnar & Zuidema)."""
        if not layer_weights:
            raise AttentionError("rollout requires at least one layer")
        n = layer_weights[0].shape[-1]
        identity = torch.eye(n, device=layer_weights[0].device)
        result: torch.Tensor | None = None
        for weights in layer_weights:
            if self.config.include_residual:
                matrix = 0.5 * weights + 0.5 * identity
            else:
                matrix = weights
            result = matrix if result is None else result @ matrix
        assert result is not None
        return result

    # ------------------------------------------------------------------ #
    # Explain
    # ------------------------------------------------------------------ #

    def explain(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        kind: str = "crop",
        class_index: int | None = None,
    ) -> dict[str, Any]:
        """Temporal importance for one sample.

        Returns a dict with ``importance`` (per timestep), ``ranking``,
        ``rollout``, ``layer_attention``, ``valid_timesteps``.
        """
        names = self._install()
        batch = single_sample_batch(sample, self.device)
        try:
            # Attention weights are captured during the forward pass.
            self.model(batch)
        finally:
            layer_weights = self._layer_weights()

        rolled = self.rollout(layer_weights)  # [B, T', T']
        clus_row = rolled[0, 0]  # [T'+1] attention from CLS
        timesteps = int(clus_row.shape[0]) - 1  # exclude CLS self-attention

        mask = to_numpy(batch.get("temporal_mask", torch.ones_like(torch.zeros(1, timesteps)))[0])
        importance = to_numpy(clus_row[1:])  # [T]
        # Zero-out padded observations.
        if mask.shape[0] == timesteps:
            importance = importance * mask

        ranking = list(np.argsort(-importance))
        agg = self.config.head_aggregation

        result = {
            "importance": importance,
            "ranking": ranking,
            "timesteps": timesteps,
            "valid_timesteps": [int(t) for t in range(timesteps) if mask[t] > 0.5],
            "rollout": to_numpy(rolled[0]),
            "layer_attention": [
                to_numpy(w[0]) for w in layer_weights
            ],
            "head_aggregation": agg,
            "layer_names": names,
        }
        return result

    # ------------------------------------------------------------------ #
    # Head-wise maps (optional, slower)
    # ------------------------------------------------------------------ #

    def head_attention(
        self, sample: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Per-head attention maps for the temporal transformer."""
        names = self._install()
        batch = single_sample_batch(sample, self.device)
        try:
            self.model(batch)
        finally:
            maps = {
                name: to_numpy(self.capture.weights(name, headwise=True))
                for name in names
            }
        return maps
