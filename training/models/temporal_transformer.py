"""Temporal transformer — encodes a variable-length fused image sequence.

Turns the fused per-timestep image features ``[B, T, D_fuse]`` into a single
``[B, embedding_dim]`` image embedding:

* learned / sinusoidal temporal positional encoding,
* an optional CLS token pooled for the embedding,
* multi-head self-attention with LayerNorm + residual + feed-forward blocks,
* an attention key-padding mask derived from the Phase 4 ``temporal_mask`` so
  padded / missing observations never contribute.

The sequence length may vary per batch (up to ``max_len``); the mask decides
which timesteps are real (``1``) versus padding (``0``).
"""

from __future__ import annotations

import torch
from torch import nn

from .config import TemporalModelConfig
from .exceptions import ShapeMismatchError
from .utils import build_key_padding_mask, build_positional_encoding, get_activation, masked_mean


class TemporalTransformer(nn.Module):
    """Temporal transformer encoder over fused image features.

    Args:
        config: Validated :class:`TemporalModelConfig`.
        input_dim: Width of each fused per-timestep feature vector
            (the :class:`~ai.models.image_fusion.ImageFusion` output width).

    Attributes:
        output_dim: Width of the image embedding (``config.embedding_dim``).
    """

    def __init__(self, config: TemporalModelConfig, input_dim: int) -> None:
        super().__init__()
        self.config = config
        self.max_len = config.max_len
        self.use_cls = config.use_cls

        self.input_proj = nn.Linear(input_dim, config.d_model)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, config.d_model)) if config.use_cls else None
        )
        self.pos_encoding = build_positional_encoding(
            config.position_encoding, config.d_model, config.max_len
        )

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=config.ff_dim,
                    dropout=config.dropout,
                    activation=get_activation(config.activation),
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.output_proj = nn.Linear(config.d_model, config.embedding_dim)
        self.output_dim = config.embedding_dim
        #: Recompute block activations during backprop (saves memory, costs
        #: compute). Only applied while training.
        self.gradient_checkpointing = False
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    # -- nn.Module ---------------------------------------------------------- #

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable / disable activation checkpointing on the transformer stack."""
        self.gradient_checkpointing = bool(enabled)

    def forward(  # type: ignore[override]
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a fused image sequence into an image embedding.

        Args:
            sequence: ``[B, T, D_fuse]`` fused per-timestep features.
            mask: ``[B, T]`` validity mask (1=real, 0=padding) as produced by
                Phase 4, or ``None`` for fully-valid sequences.

        Returns:
            ``[B, embedding_dim]`` image embedding.
        """
        if sequence.dim() != 3:
            raise ShapeMismatchError(
                f"temporal transformer expects [B, T, D], got {tuple(sequence.shape)}"
            )
        batch, timesteps, _ = sequence.shape
        if timesteps > self.max_len:
            raise ShapeMismatchError(
                f"sequence length {timesteps} exceeds temporal max_len "
                f"{self.max_len}"
            )

        x = self.input_proj(sequence)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        if self.use_cls:
            cls = self.cls_token.expand(batch, -1, -1)
            x = torch.cat([cls, x], dim=1)  # [B, T + 1, D]

        key_padding = build_key_padding_mask(mask, has_cls=self.use_cls)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, src_key_padding_mask=key_padding,
                    use_reentrant=False,
                )
            else:
                x = block(x, src_key_padding_mask=key_padding)
        x = self.norm(x)

        if self.use_cls:
            pooled = x[:, 0]
        else:
            pooled = masked_mean(x, mask, dim=1)

        return self.output_proj(self.dropout(pooled))
