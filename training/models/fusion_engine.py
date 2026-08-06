"""CrossModalFusionEngine — the complete cross-modal pathway as one unit.

The engine owns every step that turns the modality embeddings into the shared
multimodal representation:

1. **Cross attention** — image attends to tabular (``Q=image, K=V=tabular``),
2. **Adaptive gated fusion** — per-sample image / tabular / fusion gates
   (optionally a fourth temporal stream),
3. **Shared multimodal encoder** — a CLS-pooled transformer stack producing the
   final ``[B, out_dim]`` shared representation consumed by every task head.

Encapsulating the whole pathway makes the fusion strategy a swappable unit
(ablate cross-attention, gating, residual fusion or the temporal stream through
config only) and lets the :class:`~ai.models.cropfusion.CropFusionModel`
delegate all cross-modal logic here. The engine stays a pure building block:
it never touches training / loss / optimisation state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .adaptive_gate import AdaptiveGatedFusion
from .config import ModelConfig
from .cross_attention import CrossAttention
from .exceptions import ModelConfigurationError
from .shared_encoder import SharedMultimodalEncoder


@dataclass
class FusionOutput:
    """Typed result of :class:`CrossModalFusionEngine.forward`.

    ``shared_embedding`` is the final ``[B, out_dim]`` multimodal
    representation; the rest are per-stage tensors kept for explainability /
    ablations.
    """

    shared_embedding: torch.Tensor
    fused: torch.Tensor | None = None
    cross_output: torch.Tensor | None = None
    image_token: torch.Tensor | None = None
    tabular_token: torch.Tensor | None = None
    temporal_token: torch.Tensor | None = None
    gates: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shared_embedding": self.shared_embedding,
            "fused": self.fused,
            "cross_output": self.cross_output,
            "image_token": self.image_token,
            "tabular_token": self.tabular_token,
            "temporal_token": self.temporal_token,
            "gates": self.gates,
        }


class CrossModalFusionEngine(nn.Module):
    """Cross-attention + gated fusion + shared encoder (the fusion unit).

    Args:
        config: Validated :class:`ModelConfig` (cross_attention / gated_fusion
            / fusion / shared_encoder sections).
        image_dim: Width of the image embedding (temporal transformer output).
        tabular_dim: Width of the tabular embedding.

    Raises:
        ModelConfigurationError: When called without both modalities (the
            engine is multimodal-only; single-modality models use the plain
            shared encoder).
    """

    def __init__(
        self, config: ModelConfig, *, image_dim: int, tabular_dim: int
    ) -> None:
        super().__init__()
        if image_dim <= 0 or tabular_dim <= 0:
            raise ModelConfigurationError(
                "CrossModalFusionEngine requires both an image and a tabular "
                "stream (image_dim > 0 and tabular_dim > 0)",
                detail={"image_dim": image_dim, "tabular_dim": tabular_dim},
            )
        self.config = config
        cross_cfg = config.cross_attention
        gated_cfg = config.gated_fusion
        fusion_cfg = config.fusion
        shared_cfg = config.shared_encoder

        #: Temporal stream width (when enabled it equals the image embedding
        #: width — the temporal transformer output).
        temporal_dim = image_dim if fusion_cfg.use_temporal_stream else 0

        # -- 1. Cross attention ---------------------------------------------- #
        self.cross_attention: CrossAttention | None = None
        self.cross_dim = image_dim
        if cross_cfg.enabled:
            self.cross_attention = CrossAttention(
                query_dim=image_dim,
                key_dim=tabular_dim,
                num_heads=cross_cfg.num_heads,
                out_dim=cross_cfg.out_dim,
                dropout=cross_cfg.dropout,
            )
            self.cross_dim = cross_cfg.out_dim

        # -- 2. Adaptive gated fusion ---------------------------------------- #
        self.gated_fusion: AdaptiveGatedFusion | None = None
        if gated_cfg.enabled:
            self.gated_fusion = AdaptiveGatedFusion(
                image_dim=image_dim,
                tabular_dim=tabular_dim,
                cross_dim=self.cross_dim,
                out_dim=gated_cfg.out_dim,
                hidden_dim=gated_cfg.hidden_dim,
                dropout=gated_cfg.dropout,
                activation=shared_cfg.activation,
                temporal_dim=temporal_dim,
            )
            shared_input_dim = gated_cfg.out_dim
        else:
            # No adaptive gate -> concatenate the available branches.
            shared_input_dim = image_dim + tabular_dim
            if cross_cfg.enabled:
                shared_input_dim += cross_cfg.out_dim

        # -- 3. Shared multimodal encoder ------------------------------------- #
        self.shared_encoder = SharedMultimodalEncoder(shared_input_dim, shared_cfg)
        self.output_dim = self.shared_encoder.output_dim

        self.residual_fusion = bool(fusion_cfg.residual_fusion) and gated_cfg.enabled
        if self.residual_fusion:
            self.residual_norm = nn.LayerNorm(gated_cfg.out_dim)
            self.residual_dropout = nn.Dropout(gated_cfg.dropout)

    # -- nn.Module ----------------------------------------------------------- #

    def forward(
        self,
        image_embedding: torch.Tensor,
        tabular_embedding: torch.Tensor,
        temporal_embedding: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> FusionOutput:
        """Build the shared multimodal representation.

        Args:
            image_embedding: ``[B, D_img]`` temporal image embedding.
            tabular_embedding: ``[B, D_tab]`` tabular embedding.
            temporal_embedding: Optional ``[B, D_img]`` raw temporal stream
                used by the gated fusion when ``fusion.use_temporal_stream``.
            return_attention: Also capture the cross-attention weight matrix
                (requires ``cross_attention.enabled``).

        Returns:
            :class:`FusionOutput`.
        """
        cross_output: torch.Tensor = image_embedding
        cross_weights: torch.Tensor | None = None
        if self.cross_attention is not None:
            if return_attention:
                cross_output, cross_weights = self.cross_attention(
                    image_embedding, tabular_embedding, return_attention=True
                )
            else:
                cross_output = self.cross_attention(
                    image_embedding, tabular_embedding
                )

        if self.gated_fusion is not None:
            gated = self.gated_fusion(
                image_embedding,
                tabular_embedding,
                cross_output,
                temporal_embedding=temporal_embedding,
            )
            fused = gated["fused"]
            if self.residual_fusion:
                residual = gated["image_token"] + gated["tabular_token"]
                if "temporal_token" in gated:
                    residual = residual + gated["temporal_token"]
                fused = self.residual_dropout(self.residual_norm(fused + residual))
            gates = {
                key: gated[key]
                for key in (
                    "image_gate",
                    "tabular_gate",
                    "temporal_gate",
                    "fusion_gate",
                )
                if key in gated
            }
            if cross_weights is not None:
                gates["cross_attention"] = cross_weights
            shared = self.shared_encoder(fused)
            return FusionOutput(
                shared_embedding=shared,
                fused=fused,
                cross_output=cross_output,
                image_token=gated["image_token"],
                tabular_token=gated["tabular_token"],
                temporal_token=gated.get("temporal_token"),
                gates=gates,
            )

        # No adaptive gate -> concatenate the available branches.
        parts = [image_embedding, tabular_embedding]
        if self.cross_attention is not None:
            parts.append(cross_output)
        shared = self.shared_encoder(torch.cat(parts, dim=-1))
        gates: dict[str, torch.Tensor] = {}
        if cross_weights is not None:
            gates["cross_attention"] = cross_weights
        return FusionOutput(
            shared_embedding=shared,
            fused=None,
            cross_output=cross_output,
            gates=gates,
        )
