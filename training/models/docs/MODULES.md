# Module documentation

All modules live under `ai/models/`. `*` marks modules added beyond the Phase 5
file list to keep the architecture clean.

| File | Responsibility |
|------|----------------|
| `__init__.py` | Public API surface. |
| `config.py` | `ModelConfig` + per-section pydantic configs; YAML loader + template; `from_preprocessor` derivation. |
| `interfaces.py` | Ports: `ImageEncoder`, `Head`, `TaskLoss`. |
| `utils.py` | Parameter/memory accounting, `model_summary`, activations, positional encodings, mask helpers, freezing/filtering. |
| `validators.py` | `validate_batch`, `expected_batch_shapes`, `validate_model_config`. |
| `exceptions.py` * | `ModelError` hierarchy (`MDL-*`). |
| `backbone.py` * | `TimmImageEncoder` — shared timm adapter (single channel → features). |
| `ndvi_encoder.py` | `NdviEncoder`. |
| `evi_encoder.py` | `EviEncoder`. |
| `tabtransformer.py` | `TabTransformer`, `CategoricalEmbedding`, `ContinuousEmbedding`. |
| `image_fusion.py` | `ImageFusion` (concat / weighted_sum / learnable / attention). |
| `temporal_transformer.py` | `TemporalTransformer` (CLS + temporal pos encoding + masks). |
| `cross_attention.py` | `CrossAttention` (Q=image, K=V=tabular). |
| `adaptive_gate.py` | `AdaptiveGatedFusion` (image / tabular / fusion gates). |
| `shared_encoder.py` | `SharedMultimodalEncoder` (shared latent space). |
| `multitask_heads.py` | `CropHead`, `YieldHead`, `MultiTaskHeads` (registry). |
| `losses.py` | Loss interfaces: CE, label smoothing, focal, MSE, Huber, weighted multi-task. |
| `cropfusion.py` | `CropFusionModel` + `CropFusionOutput`. |
| `factory.py` | `ModelFactory` (create / from-config / from-preprocessor / freeze / load / save-config). |
| `checkpoint.py` | `CheckpointManager` (save / load / resume / partial) + `LoadReport` / `ResumeState`. |
| `exporter.py` | `ModelExporter` (TorchScript / ONNX / future TensorRT). |
| `pyproject.toml` | Package metadata + pytest config. |
| `README.md`, `docs/*` | Documentation. |
| `tests/` | Test suite. |

## Dependencies between modules

```
config ─► validators
  ▲          ▲
  │          │
factory ──► cropfusion ──► tabtransformer, ndvi_encoder, evi_encoder,
  │              │           image_fusion, temporal_transformer,
  │              │           cross_attention, adaptive_gate,
  │              │           shared_encoder, multitask_heads
  │              │
  └── checkpoint ┘
  └── exporter
interfaces ◄── backbone, tabtransformer, multitask_heads, losses
utils ◄── everything (shared building blocks)
```

Every module depends only on `utils` / `interfaces` / `config` / `exceptions` —
the standard Clean-Architecture dependency direction.
