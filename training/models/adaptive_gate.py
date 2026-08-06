"""Adaptive gated fusion — learns how much each modality contributes.

Instead of a fixed concatenation, three learnable gates decide, per sample,
how much of the shared representation comes from imagery and how much from
tabular data:

* **image gate** — scales the projected image stream,
* **tabular gate** — scales the projected tabular stream,
* **fusion gate** — blends the gated modality sum with the projected
  cross-modal attention output.

All gates are computed from the concatenated context (image + tabular +
cross-modal) of each sample, so the model adapts its trust in each modality
sample-by-sample — exactly the behaviour required for the explainability
"modality weights" output.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .utils import get_activation


class AdaptiveGatedFusion(nn.Module):
    """Learnable per-sample gating over image / tabular / cross-modal streams.

    Args:
        image_dim: Width of the image embedding.
        tabular_dim: Width of the tabular embedding.
        cross_dim: Width of the cross-attention output.
        out_dim: Width of the fused representation.
        hidden_dim: Width of the gate MLPs.
        dropout: Dropout on the fused output.
        activation: Activation used inside the gate MLPs.
        temporal_dim: Width of an optional fourth (temporal) stream. When
            ``> 0`` the forward also accepts a ``temporal_embedding`` and
            learns a dedicated ``temporal_gate`` for it.
    """

    def __init__(
        self,
        image_dim: int,
        tabular_dim: int,
        cross_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        temporal_dim: int = 0,
    ) -> None:
        super().__init__()
        act = get_activation(activation)
        self.temporal_dim = int(temporal_dim)
        context_dim = image_dim + tabular_dim + cross_dim
        if self.temporal_dim > 0:
            context_dim += self.temporal_dim

        self.image_proj = nn.Linear(image_dim, out_dim)
        self.tabular_proj = nn.Linear(tabular_dim, out_dim)
        self.temporal_proj = (
            nn.Linear(self.temporal_dim, out_dim) if self.temporal_dim > 0 else None
        )

        #: image_gate, tabular_gate [, temporal_gate] — computed jointly from
        #: the full context.
        gate_outputs = 3 if self.temporal_dim > 0 else 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, gate_outputs),
        )
        #: fusion_gate — how much to trust the cross-modal branch.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1),
        )
        #: Projection of the full context for the fusion-gated branch.
        self.context_proj = nn.Linear(context_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    # -- nn.Module ---------------------------------------------------------- #

    def forward(  # type: ignore[override]
        self,
        image_embedding: torch.Tensor,
        tabular_embedding: torch.Tensor,
        cross_output: torch.Tensor,
        temporal_embedding: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Fuse the streams under learnable per-sample gates.

        Args:
            image_embedding: ``[B, D_img]``.
            tabular_embedding: ``[B, D_tab]``.
            cross_output: ``[B, D_cross]`` cross-attention output.
            temporal_embedding: Optional ``[B, D_temporal]`` temporal stream
                (only consumed when ``temporal_dim > 0``).

        Returns:
            A dict with:

            * ``fused`` — ``[B, out_dim]`` gated fusion output,
            * ``image_gate`` / ``tabular_gate`` / ``fusion_gate`` (and
              ``temporal_gate`` when enabled) — ``[B, 1]`` gate values in
              ``[0, 1]`` (for explainability),
            * ``image_token`` / ``tabular_token`` (and ``temporal_token`` when
              enabled) — ``[B, out_dim]`` projected modality streams (fed to
              the shared encoder as context tokens).
        """
        context_parts = [image_embedding, tabular_embedding, cross_output]
        if self.temporal_dim > 0 and temporal_embedding is not None:
            context_parts.append(temporal_embedding)
        context = torch.cat(context_parts, dim=-1)

        gates = torch.sigmoid(self.gate_mlp(context))  # [B, 2] or [B, 3]
        image_gate = gates[:, 0:1]
        tabular_gate = gates[:, 1:2]
        fusion_gate = torch.sigmoid(self.fusion_mlp(context))  # [B, 1]

        image_token = self.image_proj(image_embedding)
        tabular_token = self.tabular_proj(tabular_embedding)
        gated_sum = image_gate * image_token + tabular_gate * tabular_token

        result: dict[str, Any] = {
            "image_gate": image_gate,
            "tabular_gate": tabular_gate,
            "fusion_gate": fusion_gate,
            "image_token": image_token,
            "tabular_token": tabular_token,
        }
        if self.temporal_dim > 0 and temporal_embedding is not None:
            temporal_gate = gates[:, 2:3]
            temporal_token = self.temporal_proj(temporal_embedding)
            gated_sum = gated_sum + temporal_gate * temporal_token
            result["temporal_gate"] = temporal_gate
            result["temporal_token"] = temporal_token

        context_out = self.context_proj(context)
        fused = (1.0 - fusion_gate) * gated_sum + fusion_gate * context_out
        result["fused"] = self.norm(self.dropout(fused))
        return result
