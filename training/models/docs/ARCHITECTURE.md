# Architecture

## Complete neural architecture

```
                 Tabular Data                     NDVI Sequence                EVI Sequence
                      │                                  │                            │
              TabTransformer                     EfficientNetV2-S            EfficientNetV2-S
                      │                          (timm, classifier removed)  (timm, classifier removed)
            Tabular Embedding                            │                            │
                      │                             NDVI Features             EVI Features
                      │                                  └──────────┬─────────────────┘
                      │                                             ▼
                      │                                 Image Feature Fusion
                      │                                    (concat | weighted | learnable | attention)
                      │                                             │
                      │                                             ▼
                      │                               Temporal Transformer Encoder
                      │                                 (+ CLS + temporal positional encoding)
                      │                                             │
                      │                                     Image Embedding
                      │                                             │
                      └──────────────────┬──────────────────────────┘
                                         ▼
                               Cross Modal Attention
                              (Q = Image, K = V = Tabular)
                                         │
                                         ▼
                                Adaptive Gated Fusion
                          (image gate · tabular gate · fusion gate)
                                         │
                                         ▼
                        Shared Multimodal Representation
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                   Crop Recommendation         Yield Prediction
                      (softmax)                  (regression)
```

## Data flow rules

* **Only Phase 4 output is consumed.** The model never touches raw datasets;
  it reads the batch dict produced by `ai.preprocessing` (tabular, ndvi, evi,
  temporal_mask).
* **No training.** This phase ships the forward architecture, configuration,
  factory, checkpointing and export. Optimizers, schedulers and the training
  loop are Phase 6.

## Modality routing

`ModelConfig.uses_tabular` / `uses_image` derive from the config:

* Both enabled (default) → full pipeline above.
* Tabular only (`image_encoder.backbone: null`) → `TabTransformer` →
  shared encoder → heads.
* Image only (empty tabular schema) → image encoders → temporal transformer →
  shared encoder → heads.

The shared encoder and heads are always present; cross-attention and gated
fusion exist only in the multimodal path.

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Pre-norm transformer blocks | Stable training at depth; standard in modern LLMs/vision. |
| CLS-token pooling | Fixed-width embedding from variable-length sequences. |
| Attention key-padding from `temporal_mask` | Padded/missing observations never contribute. |
| Learnable per-sample gates | Model decides modality trust per location/sample. |
| Probing real backbone feature width | `timm.num_features` under-reports some backbones. |
| Single `forward_export` tensor path | Clean TorchScript / ONNX tracing. |
