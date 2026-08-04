# CropFusion — Phase 5 Completion Report

**Phase:** Complete Neural Architecture (Crop Recommendation + Yield Prediction)
**Status:** ✅ Complete
**Date:** 2026-08-02
**Tests:** 131 passed (Phase 5) · **Full regression:** 432 passed (Phases 2–5)
**Coverage:** 90% (models package)

---

## ✔ Files Created

```
ai/models/
├── __init__.py             # public API
├── factory.py              # ModelFactory (create/from-config/from-preprocessor/freeze/load)
├── cropfusion.py           # CropFusionModel + CropFusionOutput
├── tabtransformer.py       # TabTransformer + Categorical/ContinuousEmbedding
├── backbone.py             # TimmImageEncoder (shared timm adapter)
├── ndvi_encoder.py         # NdviEncoder
├── evi_encoder.py          # EviEncoder
├── image_fusion.py         # ImageFusion (concat/weighted_sum/learnable/attention)
├── temporal_transformer.py # TemporalTransformer (CLS + temporal pos + masks)
├── cross_attention.py      # CrossAttention (Q=image, K=V=tabular)
├── adaptive_gate.py        # AdaptiveGatedFusion (image/tabular/fusion gates)
├── shared_encoder.py       # SharedMultimodalEncoder (shared latent space)
├── multitask_heads.py      # CropHead / YieldHead / MultiTaskHeads (registry)
├── losses.py               # Loss interfaces (CE, label-smoothing, focal, MSE, Huber, weighted multi-task)
├── checkpoint.py           # CheckpointManager (save/load/resume/partial)
├── exporter.py             # ModelExporter (TorchScript / ONNX / future TensorRT)
├── config.py               # ModelConfig (+ per-section pydantic configs, YAML loader)
├── interfaces.py           # ports (ImageEncoder / Head / TaskLoss)
├── utils.py                # params/memory/summary, activations, positional encodings, masks
├── validators.py           # batch/config validation + expected shapes
├── exceptions.py           # ModelError hierarchy (MDL-*)
├── pyproject.toml
├── README.md
├── docs/  (ARCHITECTURE, LAYER, TENSOR_FLOW, MODULES, CONFIGURATION, DEVELOPER)
└── tests/  (16 test modules + conftest)
```

*`backbone.py` and `exceptions.py` are additions beyond the listed file set to
keep the shared timm adapter and error hierarchy DRY (matching the convention
of every other CropFusion package).*

## ✔ Classes

| Class | Responsibility |
|-------|----------------|
| `CropFusionModel` | Full architecture: forward / forward_export / summary / add_head |
| `TabTransformer` | Categorical + continuous tokens, CLS, pre-norm transformer blocks |
| `CategoricalEmbedding` / `ContinuousEmbedding` | Token embeddings (OOV slot reserved) |
| `TimmImageEncoder` | timm backbone adapter: `[B,T,1,H,W]` → `[B,T,D]` (probing feature width) |
| `NdviEncoder` / `EviEncoder` | Independent NDVI / EVI encoders |
| `ImageFusion` | Per-timestep fusion: concat / weighted_sum / learnable / attention |
| `TemporalTransformer` | Variable-length sequence encoder, CLS pooling, temporal pos encoding |
| `CrossAttention` | Q = image embedding, K = V = tabular embedding |
| `AdaptiveGatedFusion` | Per-sample image / tabular / fusion gates |
| `SharedMultimodalEncoder` | Shared latent space (configurable 512/768/1024) |
| `CropHead` / `YieldHead` / `MultiTaskHeads` | Task heads + extensible registry |
| `CrossEntropyLoss` / `LabelSmoothingLoss` / `FocalLoss` | Crop classification losses |
| `MSELoss` / `HuberLoss` | Yield regression losses |
| `WeightedMultiTaskLoss` | Fixed or learnable (Kendall) multi-task weighting |
| `CheckpointManager` (+`LoadReport`/`ResumeState`) | Save / load / resume / partial loading |
| `ModelExporter` | TorchScript (trace) / ONNX / future TensorRT |
| `ModelFactory` | Construction, config files, preprocessor derivation, freezing, pretrained loading |
| `ModelConfig` (+ per-section configs) | Everything configurable, YAML + env overrides |

## ✔ Architecture (as specified)

```
tabular [B,F]  ──►  TabTransformer  ──►  tabular embedding
ndvi [B,T,1,H,W] ──► NdviEncoder (timm) ─┐
evi  [B,T,1,H,W] ──► EviEncoder  (timm) ─┴─► ImageFusion ──► TemporalTransformer ──► image embedding
image embedding + tabular embedding ──► CrossAttention (Q=image, K=V=tabular)
    ──► AdaptiveGatedFusion (image·tabular·fusion gates)
    ──► SharedMultimodalEncoder ──► CropHead (softmax) + YieldHead (regression)
```

## ✔ Parameter summary

Default configuration (`efficientnetv2_s`, crop classes = 20, tabular
12 numeric + 3 categorical):

| Component | Parameters |
|-----------|-----------:|
| TabTransformer | 206 K |
| NdviEncoder (EfficientNetV2-S) | 20.2 M |
| EviEncoder (EfficientNetV2-S) | 20.2 M |
| ImageFusion (learnable, capped 512) | 5.0 M |
| TemporalTransformer | 1.8 M |
| CrossAttention | 0.3 M |
| AdaptiveGatedFusion | 0.5 M |
| SharedMultimodalEncoder | 1.8 M |
| Multi-task heads | 0.5 M |
| **Total** | **≈ 50.5 M** |

`CropFusionModel.summary()` reports parameter counts, a recursive layer
summary and memory estimates (parameters + peak activation) for any config.

## ✔ Configuration options

YAML (or `MODEL_CONFIG_FILE` + `MODEL_<SECTION>__<KEY>` env) covers:
`tabular` (numeric_dim, categorical_cardinalities, embedding_dim, depth,
heads, ff_dim, dropout, activation, use_cls, position_encoding, max_len) ·
`image_encoder` (backbone, pretrained, freeze_backbone, input_size,
channel_expansion, drop_path_rate, ndvi/evi backbone overrides) ·
`image_fusion` (method, hidden_dim, dropout) · `temporal` (d_model, depth,
heads, ff_dim, dropout, activation, use_cls, position_encoding, max_len,
embedding_dim) · `cross_attention` (heads, dropout, out_dim) ·
`gated_fusion` (out_dim, hidden_dim, dropout) · `shared_encoder` (d_model,
depth, heads, ff_dim, dropout, activation, out_dim) · `heads` (crop.num_classes,
yield clamp, hidden/dropout/activation) · `loss` (crop/yield loss, weights,
weighting mode, label smoothing, focal gamma) · `checkpoint` · `export` ·
`validate_inputs`.

`ModelConfig.from_preprocessor(preprocessor)` derives the tabular schema,
class count and image size automatically; `save_model_template()` writes an
annotated default YAML.

## ✔ Integration points

* **Phase 4** — the model consumes the exact
  `ai.preprocessing.dataloader.collate_samples` batch: `tabular [B,F]`,
  `ndvi/evi [B,T,1,H,W]`, `temporal_mask [B,T]`. `ModelFactory.from_preprocessor`
  builds a model from a fitted Phase 4 `Preprocessor` (ordinal categorical
  schema → TabTransformer; one-hot → all-continuous).
* **Phase 6 (training)** — `forward` returns typed `CropFusionOutput`;
  `losses.WeightedMultiTaskLoss(inputs, targets)` consumes it;
  `CheckpointManager.resume` restores model + optimizer + scheduler state.
* **Serving/export** — `forward_export` returns a plain tensor tuple;
  `ModelExporter` traces TorchScript / ONNX (dynamic batch + time axes).
* **Explainability** — per-sample `image_gate` / `tabular_gate` /
  `fusion_gate` from `AdaptiveGatedFusion` feed the SDD "modality weights".

## ✔ Future extensions

* Phase 6 training loop, optimizers, schedulers, metrics, early stopping.
* TensorRT engine build (entry point + validation already in place).
* Per-index temporal validity masks (`[T, 2]`) from Phase 4.
* Additional task heads (crop health / disease / water requirement) via
  `model.add_head(...)`.
* ONNX export test coverage once `onnx` is installed.

## ✔ Known limitations

* **No training code** — losses are interfaces only; optimizers/schedulers are
  Phase 6 (per the phase boundary).
* **ONNX export is implemented but untested** in this environment — the `onnx`
  package is not installed, so the success path and TensorRT are gated behind
  `MissingDependencyError` (documented, not stubbed).
* **Ordinal encoding required** for categorical embeddings — Phase 4 with
  one-hot encoding is consumed as an all-continuous vector.
* **Tracing-based TorchScript** — the dict-based `forward` is not scriptable;
  `forward_export` is the traced entry point.
* BatchNorm backbones need batch ≥ 2 in train mode; the construction probe and
  exporter run in eval mode.
* The timm feature width is **probed** at construction (`num_features` can
  under-report, e.g. `mobilenetv3_small_050`), adding one forward pass at
  build time.

---

## Phase boundary

Phases 2 (Dataset Manager), 3 (STAM), 4 (Preprocessing) and 5 (AI architecture)
are **complete** and verified (432 tests green). No training, optimizer,
scheduler or inference API has been implemented. Per instructions I am
**stopping** — Phase 6 (training) has not begun.

**Awaiting:** `"Proceed to Phase 6"`
