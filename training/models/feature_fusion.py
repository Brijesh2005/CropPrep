"""Feature-level fusion models (the CropFusion "feature fusion" architecture).

The central hypothesis of this architecture is:

    GOOD UNIMODAL REPRESENTATIONS
        + FEATURE-LEVEL FUSION
        + MASKED TEMPORAL AGGREGATION
        + PROPER FIELD-LEVEL SPLITS
        =
    A SIMPLER, MORE ROBUST CROP MODEL

instead of making one huge multimodal Transformer reason over every raw
modality simultaneously.

Modality encoders
-----------------
* ``TabularFeatureEncoder`` — environmental tabular -> 256-D embedding
  (Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm).
* ``ImageFeatureEncoder``   — per-observation feature projector maps each
  satellite-composite observation (or image patch embedding) to 512-D.
  NOTE ON THE LOCAL IMAGERY MODALITY: raw Sentinel-2 patches are only
  available on Kaggle; the legitimate locally-reproducible "imagery"
  representation is the per-year DK grid vegetation-composite feature vector
  (NDVI/EVI/NDWI/NDRE/SAVI/S2 counts + Kharif/Rabi composites). Each grid-year
  observation is a 12-D vector and is projected by an MLP. On a platform with
  real image patches, the SAME interface accepts per-patch CNN embeddings
  (e.g. EfficientNet pooling); ``image_feature_dim`` then becomes the CNN
  embedding width and the projector can be dropped.
* ``TemporalAttentionPool`` — masked attention pooling over [T, D] -> [D]:
  attention scores from an MLP, hard mask of padded timestamps, softmax over
  valid timestamps, weighted sum. Padded (unobserved) timestamps receive
  exactly zero weight, so they never influence the field-level representation.

Fusion
------
* ``FeatureFusion`` — feature-level fusion of the two embeddings:
  ``concat`` (default): concat[tabular | image] -> MLP -> 256-D fused vector.
  ``gated``: per-modality sigmoid gates applied before concatenation.

Classifier head
---------------
* ``FeatureFusionClassifier`` — end-to-end model exposing logits + embeddings.
  Binary head (BCEWithLogitsLoss-compatible), num_classes=1.

Everything is deliberately simple and modular for easy debugging.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _sequential(blocks: list[tuple[int, int]], dropout: float,
                activation: str = "gelu") -> nn.Sequential:
    act = nn.GELU() if activation == "gelu" else nn.ReLU()
    layers: list[nn.Module] = []
    for i, (in_f, out_f) in enumerate(blocks):
        layers.append(nn.Linear(in_f, out_f))
        if i < len(blocks) - 1:
            layers.append(nn.LayerNorm(out_f))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TabularFeatureEncoder(nn.Module):
    """Environmental tabular features -> 256-D tabular embedding."""

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 embedding_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.net = _sequential(
            [(input_dim, hidden_dim), (hidden_dim, embedding_dim)],
            dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageFeatureEncoder(nn.Module):
    """Per-observation feature encoder -> [B, T, image_embedding_dim].

    ``image_feature_dim`` is either the raw composite-feature width (local DK
    grid representation) or the width of a pretrained CNN patch embedding on a
    platform with real image patches. Each observation is projected
    independently (no cross-observation leakage).
    """

    def __init__(self, image_feature_dim: int, hidden_dim: int = 128,
                 embedding_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projector = _sequential(
            [(image_feature_dim, hidden_dim), (hidden_dim, embedding_dim)],
            dropout=dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.projector(seq)


class TemporalAttentionPool(nn.Module):
    """Masked attention pooling over [B, T, D] -> [B, D].

    attention_score = MLP(token_embedding)
    scores masked to -inf at padded timestamps
    softmax over valid timestamps
    weighted sum of token embeddings

    If a field has zero valid timestamps its pooled embedding is zero
    (never NaN), and padded timestamps never influence valid fields.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, seq: torch.Tensor,
                temporal_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(seq).squeeze(-1)                    # [B, T]
        scores = scores.masked_fill(temporal_mask == 0.0, -1e9)
        attn = torch.softmax(scores, dim=-1)                    # [B, T]
        # softmax subtracts the row max, so masked (-1e9) rows become uniform
        # (0.25) rather than zero; zero padded mass explicitly and renormalize
        # over the valid timestamps. Fully-masked rows -> exact zeros, no NaN.
        attn = attn * temporal_mask                             # [B, T]
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn = attn / denom
        return (attn.unsqueeze(-1) * seq).sum(dim=1)            # [B, D]


class FeatureFusion(nn.Module):
    """Feature-level fusion: tabular_256 + image_512 -> fused_256.

    fusion_type:
      concat — default, concatenate the two embeddings then MLP-fuse.
      gated  — separate per-modality sigmoid gates applied before concat.
    """

    def __init__(self, tabular_embedding_dim: int = 256,
                 image_embedding_dim: int = 512, fusion_dim: int = 256,
                 hidden_dim: int = 512, dropout: float = 0.2,
                 fusion_type: str = "concat"):
        super().__init__()
        if fusion_type not in ("concat", "gated"):
            raise ValueError(f"unknown fusion_type '{fusion_type}'")
        self.fusion_type = fusion_type
        self.tabular_dim = tabular_embedding_dim
        self.image_dim = image_embedding_dim
        self.fusion_dim = fusion_dim
        if fusion_type == "gated":
            self.tabular_gate = nn.Sequential(
                nn.Linear(tabular_embedding_dim, 64), nn.SiLU(),
                nn.Linear(64, tabular_embedding_dim), nn.Sigmoid())
            self.image_gate = nn.Sequential(
                nn.Linear(image_embedding_dim, 64), nn.SiLU(),
                nn.Linear(64, image_embedding_dim), nn.Sigmoid())
        self.fusor = _sequential(
            [(tabular_embedding_dim + image_embedding_dim, hidden_dim),
             (hidden_dim, fusion_dim)],
            dropout=dropout)

    def forward(self, tabular: torch.Tensor, image: torch.Tensor,
                ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.fusion_type == "gated":
            g_t = self.tabular_gate(tabular)
            g_i = self.image_gate(image)
            tab = tabular * g_t
            img = image * g_i
            gates = {"tabular_gate": g_t, "image_gate": g_i}
        else:
            tab, img = tabular, image
            gates = {"tabular_gate": None, "image_gate": None}
        fused = self.fusor(torch.cat([tab, img], dim=-1))
        return fused, gates


class FeatureFusionClassifier(nn.Module):
    """End-to-end feature-level fusion classifier.

    forward(tabular, images, temporal_mask) -> dict with keys:
        logits             [B, 1] binary logits
        tabular_embedding  [B, 256]
        image_embedding    [B, 512]
        fused_embedding    [B, fusion_dim]
        reasoning          small info dict
    """

    def __init__(self, tabular_dim: int, image_feature_dim: int,
                 tabular_embedding_dim: int = 256,
                 image_embedding_dim: int = 512, fusion_dim: int = 256,
                 hidden_dim: int = 512, dropout: float = 0.2,
                 fusion_type: str = "concat", num_classes: int = 1):
        super().__init__()
        self.tabular_encoder = TabularFeatureEncoder(
            tabular_dim, embedding_dim=tabular_embedding_dim, dropout=dropout)
        self.image_encoder = ImageFeatureEncoder(
            image_feature_dim, embedding_dim=image_embedding_dim)
        self.temporal_pool = TemporalAttentionPool(image_embedding_dim)
        self.fusion = FeatureFusion(
            tabular_embedding_dim=tabular_embedding_dim,
            image_embedding_dim=image_embedding_dim, fusion_dim=fusion_dim,
            hidden_dim=hidden_dim, dropout=dropout, fusion_type=fusion_type)
        self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, tabular: torch.Tensor, images: torch.Tensor,
                temporal_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        tab = self.tabular_encoder(tabular)
        img = self.image_encoder(images)                 # [B, T, E]
        img = self.temporal_pool(img, temporal_mask)     # [B, E]
        fused, gates = self.fusion(tab, img)
        logits = self.head(fused)
        return {
            "logits": logits,
            "tabular_embedding": tab,
            "image_embedding": img,
            "fused_embedding": fused,
            "reasoning": gates,
        }


class _UnimodalFeatureFusion(nn.Module):
    """Shared bare-bones wrapper: encoder -> head (no cross-modality fusion).

    Only used to build the modality-only baselines (tab-only / image-only)
    from the same encoder blocks so ablations stay apples-to-apples.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 256,
                 hidden_dim: int = 512, dropout: float = 0.2,
                 image: bool = False, temporal: bool = False):
        super().__init__()
        self.image = image
        self.temporal = temporal
        if image:
            self.encoder = ImageFeatureEncoder(input_dim, embedding_dim=embedding_dim)
            self.temporal_pool = TemporalAttentionPool(embedding_dim)
        else:
            self.encoder = TabularFeatureEncoder(input_dim, embedding_dim=embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim))
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor,
                temporal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.image:
            emb = self.encoder(x)                                  # [B, T, D]
            emb = self.temporal_pool(emb, temporal_mask)           # [B, D]
        else:
            emb = self.encoder(x)                                  # [B, D]
        logits = self.head(self.decoder(emb))
        return {"logits": logits, "embedding": emb}


def tabular_only_classifier(input_dim: int, **kw) -> nn.Module:
    return _UnimodalFeatureFusion(input_dim, image=False, **kw)


def imagery_only_classifier(image_feature_dim: int, **kw) -> nn.Module:
    kw.setdefault("embedding_dim", 512)
    return _UnimodalFeatureFusion(image_feature_dim, image=True,
                                  temporal=True, **kw)