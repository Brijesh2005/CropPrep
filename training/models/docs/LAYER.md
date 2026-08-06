# Layer diagram

The default configuration (see [CONFIGURATION.md](CONFIGURATION.md)) gives the
following widths. `D_enc` is the probed timm backbone width.

```
                       Tabular (numeric + categorical)
                                    │
   ┌────────────────────────────────▼─────────────────────────────────┐
   │ TabTransformer                                                     │
   │   categorical → CategoricalEmbedding [D]   (per feature, OOV=0)   │
   │   continuous  → ContinuousEmbedding [D]    (single token)        │
   │   tokens + CLS → N× {MHA + FFN} (pre-norm, residual, dropout)    │
   │   → LayerNorm → output_proj                                      │
   └────────────────────────────────┬─────────────────────────────────┘
                                    │  tabular embedding [B, D_tab]
                                    │
   ndvi [B,T,1,H,W]                 │                 evi [B,T,1,H,W]
        │                           │                        │
   ┌────▼───────────────────┐       │             ┌──────────▼───────────┐
   │ NdviEncoder (timm)     │       │             │ EviEncoder (timm)    │
   │  1→3ch, resize, F(x)   │       │             │  1→3ch, resize, F(x) │
   │  → [B,T,D_enc]         │       │             │  → [B,T,D_enc]       │
   └────┬───────────────────┘       │             └──────────┬───────────┘
        │                          │                        │
        └──────────────┬───────────┴────────────────────────┘
                       ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ ImageFusion  (concat | weighted_sum | learnable | attention)       │
   │   → [B, T, D_fuse]                                                 │
   └────────────────────────────────┬──────────────────────────────────┘
                                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ TemporalTransformer                                                │
   │   input_proj → + temporal positional encoding                     │
   │   CLS + N× {MHA + FFN}  (key-padding from temporal_mask)          │
   │   → LayerNorm → output_proj                                       │
   └────────────────────────────────┬──────────────────────────────────┘
                                    │  image embedding [B, D_img]
                                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ CrossAttention   Q=image, K=V=tabular                             │
   │   MHA → out_proj → + residual(image) → LayerNorm                  │
   │   → [B, D_cross]                                                  │
   └────────────────────────────────┬──────────────────────────────────┘
                                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ AdaptiveGatedFusion                                               │
   │   gates = sigmoid(MLP(image ⊕ tabular ⊕ cross))                   │
   │   image_gate · image_proj + tabular_gate · tabular_proj           │
   │   fused = (1 − fusion_gate)·gated_sum + fusion_gate·ctx_proj      │
   │   → LayerNorm → Dropout → [B, D_fuse]                             │
   └────────────────────────────────┬──────────────────────────────────┘
                                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ SharedMultimodalEncoder                                           │
   │   CLS + input token → N× {MHA + FFN} → LayerNorm → output_proj    │
   │   → [B, D_shared]   (512 / 768 / 1024 ...)                        │
   └────────────────────────────────┬──────────────────────────────────┘
                                    ▼
   ┌───────────────────────────────┬───────────────────────────────────┐
   │ CropHead                       │ YieldHead                         │
   │ Linear→act→Dropout→Linear      │ Linear→act→Dropout→Linear→clamp   │
   │ → [B, num_classes] logits      │ → [B, 1] predicted yield          │
   └───────────────────────────────┴───────────────────────────────────┘
```

## Default widths

| Component | Width | Notes |
|-----------|-------|-------|
| TabTransformer embedding | 64 | `tabular.embedding_dim` |
| Backbone features `D_enc` | probed | e.g. 1280 for `efficientnetv2_s` |
| Image fusion `D_fuse` | 512 | `image_fusion.hidden_dim`, or `min(encoder width, 512)` |
| Temporal transformer `D_img` | 256 | `temporal.embedding_dim` |
| Cross attention `D_cross` | 256 | `cross_attention.out_dim` |
| Gated fusion `D_fuse` | 256 | `gated_fusion.out_dim` |
| Shared representation `D_shared` | 512 | `shared_encoder.out_dim` |

Every width is configurable; the pipeline adapts through linear projections at
each boundary.
