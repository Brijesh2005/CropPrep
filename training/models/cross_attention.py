"""Cross-modal attention: the image embedding attends to the tabular embedding.

Per the Phase 5 specification the query is the **image** embedding and the
key/value are the **tabular** embedding — the model learns which tabular
context matters for the imagery of each sample. Includes multi-head attention,
a residual connection from the query stream, LayerNorm and dropout.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttention(nn.Module):
    """Cross attention with Q = image embedding, K = V = tabular embedding.

    Args:
        query_dim: Width of the image embedding (query).
        key_dim: Width of the tabular embedding (key / value).
        num_heads: Number of attention heads.
        out_dim: Width of the cross-attention output.
        dropout: Dropout inside attention and on the output projection.
        return_attention: When ``True``, :meth:`forward` also returns the
            attention weights (used for explainability / evaluation).

    Attributes:
        output_dim: Width of the output (``out_dim``).
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        out_dim: int,
        dropout: float = 0.1,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.return_attention = return_attention
        self.output_dim = out_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(query_dim, out_dim)
        #: Residual projection of the query (image) stream into ``out_dim``.
        self.residual_proj = nn.Linear(query_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    # -- nn.Module ---------------------------------------------------------- #

    def forward(  # type: ignore[override]
        self,
        image_embedding: torch.Tensor,
        tabular_embedding: torch.Tensor,
        *,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend the image embedding to the tabular embedding.

        Args:
            image_embedding: ``[B, D_q]`` image embedding (query).
            tabular_embedding: ``[B, D_k]`` tabular embedding (key/value).
            return_attention: Override the constructor flag for this call.
                When ``True``, returns ``(output, weights)`` where weights are
                ``[B, heads, 1, 1]`` per-sample attention over the single
                tabular token.

        Returns:
            ``[B, out_dim]`` cross-attended representation. When attention is
            requested, returns ``(output, weights)``.
        """
        want_weights = (
            self.return_attention if return_attention is None else return_attention
        )
        query = image_embedding.unsqueeze(1)  # [B, 1, D_q]
        key = tabular_embedding.unsqueeze(1)  # [B, 1, D_k]
        value = tabular_embedding.unsqueeze(1)

        attended, weights = self.attention(
            query, key, value, need_weights=want_weights
        )  # [B, 1, D_q]

        attended = self.dropout(self.out_proj(attended.squeeze(1)))
        residual = self.residual_proj(image_embedding)
        out = self.norm(attended + residual)

        if want_weights:
            return out, weights
        return out
