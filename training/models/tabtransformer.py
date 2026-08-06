"""TabTransformer — the tabular branch of CropFusion.

Consumes the Phase 4 ``[F]`` tabular tensor laid out as
``[continuous (numeric_dim) | categorical ordinal codes]`` and produces a
``[B, embedding_dim]`` tabular embedding.

Per the original TabTransformer paper (Huang et al., 2020) plus the Phase 5
extension:

* every categorical feature gets a learned embedding token,
* the continuous features are projected into a single token,
* an optional CLS token is prepended and pooled for the embedding,
* a stack of pre-norm transformer blocks (with residual + dropout) mixes the
  tokens.

Phase 4's :class:`~ai.preprocessing.transforms.OrdinalEncoder` encodes unseen
categories as ``-1``; the categorical embedding reserves index 0 for OOV
(``padding_idx=0``) and shifts input codes by one so unseen values land on a
dedicated, zero-contribution slot.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import TabularModelConfig
from .exceptions import ModelConfigurationError, ShapeMismatchError
from .utils import build_positional_encoding, get_activation


class CategoricalEmbedding(nn.Module):
    """Learned embedding for one categorical feature (OOV slot reserved)."""

    def __init__(self, cardinality: int, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if cardinality < 1:
            raise ModelConfigurationError(
                "categorical cardinality must be >= 1", detail=cardinality
            )
        self.cardinality = int(cardinality)
        # index 0 is reserved for OOV / unseen categories (zero vector,
        # no gradient) — see module docstring.
        self.embedding = nn.Embedding(
            self.cardinality + 1, embedding_dim, padding_idx=0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Embed ``[B]`` ordinal codes (0..C-1, or -1 for unseen)."""
        shifted = indices.clamp(-1, self.cardinality - 1) + 1  # -1 -> 0 (OOV)
        return self.dropout(self.embedding(shifted))


class ContinuousEmbedding(nn.Module):
    """Projects the continuous feature vector into a single token."""

    def __init__(
        self,
        dim: int,
        embedding_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Linear(dim, embedding_dim)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, dim]`` continuous features to ``[B, embedding_dim]``."""
        return self.dropout(self.activation(self.proj(x)))


class TabTransformer(nn.Module):
    """Tabular branch: mixed categorical + continuous -> tabular embedding.

    Args:
        config: Validated :class:`TabularModelConfig`.

    Attributes:
        output_dim: Width of the tabular embedding (``embedding_dim``).
    """

    def __init__(self, config: TabularModelConfig) -> None:
        super().__init__()
        if config.numeric_dim == 0 and not config.categorical_cardinalities:
            raise ModelConfigurationError(
                "TabTransformer requires numeric features and/or categorical "
                "features (numeric_dim > 0 or categorical_cardinalities)"
            )
        self.config = config
        self.numeric_dim = config.numeric_dim
        self.cardinalities = list(config.categorical_cardinalities)

        self.cat_embeddings = nn.ModuleList(
            [
                CategoricalEmbedding(cardinality, config.embedding_dim, config.dropout)
                for cardinality in self.cardinalities
            ]
        )
        self.cont_embedding: ContinuousEmbedding | None = (
            ContinuousEmbedding(
                config.numeric_dim, config.embedding_dim, config.dropout,
                config.activation,
            )
            if config.numeric_dim > 0
            else None
        )

        self.use_cls = config.use_cls
        if config.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))

        self.pos_encoding = build_positional_encoding(
            config.position_encoding, config.embedding_dim, config.max_len
        )

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.embedding_dim,
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
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output_dim = config.embedding_dim

        #: Number of feature tokens (categorical + continuous) before CLS.
        self.feature_count = len(self.cardinalities) + (1 if self.cont_embedding else 0)
        #: Recompute block activations during backprop (saves memory, costs
        #: compute). Only applied while training.
        self.gradient_checkpointing = False
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.use_cls:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    # -- nn.Module ---------------------------------------------------------- #

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable / disable activation checkpointing on the transformer stack."""
        self.gradient_checkpointing = bool(enabled)

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Encode a tabular tensor into a ``[B, embedding_dim]`` embedding.

        Args:
            tabular: ``[B, F]`` float tensor, laid out as
                ``[continuous (numeric_dim) | categorical ordinal codes]``.

        Returns:
            ``[B, embedding_dim]`` tabular embedding.
        """
        if tabular.dim() != 2:
            raise ShapeMismatchError(
                f"tabular must be [B, F], got {tuple(tabular.shape)}"
            )
        expected = self.numeric_dim + len(self.cardinalities)
        if tabular.size(1) != expected:
            raise ShapeMismatchError(
                f"tabular width {tabular.size(1)} != config {expected} "
                "(numeric_dim + categorical cardinalities)"
            )

        batch = tabular.size(0)
        tokens: list[torch.Tensor] = []

        if self.cardinalities:
            cat_codes = tabular[:, self.numeric_dim:].long()  # [B, F_cat]
            for index, embedding in enumerate(self.cat_embeddings):
                tokens.append(embedding(cat_codes[:, index]))

        if self.cont_embedding is not None:
            tokens.append(self.cont_embedding(tabular[:, : self.numeric_dim]))

        x = torch.stack(tokens, dim=1)  # [B, feature_count, D]
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        if self.use_cls:
            cls = self.cls_token.expand(batch, -1, -1)
            x = torch.cat([cls, x], dim=1)  # [B, feature_count + 1, D]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x = block(x)
        x = self.norm(x)

        if self.use_cls:
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)

        return self.output_proj(self.dropout(pooled))
