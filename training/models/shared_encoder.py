"""Shared multimodal encoder — the joint latent space.

Turns the gated fusion output into a single shared representation used by all
task heads. The fused vector is projected and presented alongside a learnable
CLS token to a stack of pre-norm transformer blocks, so the shared latent is
a genuine self-attended token (configurable width — 512 / 768 / 1024 ...).
"""

from __future__ import annotations

import torch
from torch import nn

from .config import SharedEncoderConfig
from .utils import get_activation


class SharedMultimodalEncoder(nn.Module):
    """Shared latent encoder over the fused multimodal representation.

    Args:
        input_dim: Width of the gated fusion output.
        config: Validated :class:`SharedEncoderConfig`.

    Attributes:
        output_dim: Width of the shared representation (``config.out_dim``).
    """

    def __init__(self, input_dim: int, config: SharedEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(input_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
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
        self.output_proj = nn.Linear(config.d_model, config.out_dim)
        self.output_dim = config.out_dim
        #: Recompute block activations during backprop (saves memory, costs
        #: compute). Only applied while training.
        self.gradient_checkpointing = False
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    # -- nn.Module ---------------------------------------------------------- #

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable / disable activation checkpointing on the transformer stack."""
        self.gradient_checkpointing = bool(enabled)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Build the shared multimodal representation.

        Args:
            fused: ``[B, input_dim]`` gated-fusion output.

        Returns:
            ``[B, out_dim]`` shared representation.
        """
        batch = fused.size(0)
        token = self.input_proj(fused).unsqueeze(1)  # [B, 1, D]
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, token], dim=1)  # [B, 2, D]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                sequence = torch.utils.checkpoint.checkpoint(
                    block, sequence, use_reentrant=False
                )
            else:
                sequence = block(sequence)
        sequence = self.norm(sequence)

        shared = self.output_proj(self.dropout(sequence[:, 0]))
        return shared
