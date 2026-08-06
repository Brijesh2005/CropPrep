# Tensor flow

Shapes shown for a batch of size `B`, `T` timesteps, patch `H × W`, `F`
tabular features. `D_enc` is the probed timm backbone width; other `D_*` are
config widths.

## Phase 4 → model

```
collate_samples(batch)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  tabular       [B, F]          (float32; numeric then categorical)   │
│  ndvi          [B, T, 1, H, W] (float32)                            │
│  evi           [B, T, 1, H, W] (float32)                            │
│  temporal_mask [B, T]          (float32; 1 = real, 0 = padding)     │
│  crop_label    [B]             (int64)  — unused by the model       │
│  yield_label   [B]             (float32) — unused by the model      │
└──────────────────────────────────────────────────────────────────────┘
```

## Forward pass

```
tabular [B, F]
   │  TabTransformer
   ▼
tabular_embedding [B, D_tab]

ndvi [B, T, 1, H, W] ── NdviEncoder ──► ndvi_features [B, T, D_enc]
evi  [B, T, 1, H, W] ── EviEncoder  ──► evi_features  [B, T, D_enc]
   │
   └────────────► ImageFusion ──► fused_sequence [B, T, D_fuse]
                        │
                        ▼
            TemporalTransformer (mask = temporal_mask [B, T])
                        │
                        ▼
            image_embedding [B, D_img]

image_embedding ── CrossAttention(Q=image, K=V=tabular) ──► cross_output [B, D_cross]

AdaptiveGatedFusion(image_embedding [B, D_img],
                    tabular_embedding [B, D_tab],
                    cross_output [B, D_cross])
   ├──► fused [B, D_fuse]
   └──► gates {image_gate, tabular_gate, fusion_gate} [B, 1] each

fused ── SharedMultimodalEncoder ──► shared [B, D_shared]

shared ── CropHead ──► crop_logits [B, num_classes]
shared ── YieldHead ──► yield_pred [B, 1]
```

## Returned output

`CropFusionOutput` exposes `crop_logits`, `yield_pred`,
`shared_representation`, `tabular_embedding`, `image_embedding` and `gates`.

## Mask semantics inside the temporal transformer

```
temporal_mask [B, T]            key-padding mask [B, T + 1]
  1  1  0  0                        F  F  T  T        (CLS col prepended)
  ▲  ▲        ▲                       ▲        ▲
  real        padding               CLS        padding ignored
```

`True` positions are excluded from attention, so padded/missing timesteps
never contribute to the CLS image embedding.

## Export shapes

`forward_export(tabular, ndvi, evi, temporal_mask)` returns
`(crop_logits, yield_pred, shared_representation)` as a plain tuple, which is
what TorchScript / ONNX trace. Batch and time axes are dynamic in ONNX export.
