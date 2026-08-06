# Migration Report — R2.3 → R2.4 (AI Architecture: Fusion Engine & Model Runtime)

Detailed record of the R2.4 phase: completing the **AI architecture** layer that
R2.3's datasets will feed and the Phase 6 training loop will consume — the
`CrossModalFusionEngine`, model-runtime support (precision / AMP / compile /
parallelism / gradient checkpointing), an architecture registry + version
management, richer model summary/metadata, and checkpoint-embedded architecture
identity. Companion to [MIGRATION_REPORT_R2.3](MIGRATION_REPORT_R2.3.md).

- **Status**: complete
- **Date**: 2026-08-06
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Architecture only**: this phase implements the AI architecture surface — no
  training loop, losses, optimisation, evaluation or prediction platform. Those
  remain Phase 6+.
- **Backward-compatible extensions**: no existing `training/models` test was
  modified; every existing public attribute (`model.tab_encoder`,
  `model.shared_encoder`, `model.heads`, `model.gates`, `forward_export`, the
  `crop_logits` / `yield_pred` / `shared_representation` outputs) still works.
  New behaviour is additive (new config sections, new engine, new helpers).
- **Reuse, not re-implementation**: the fusion engine composes the existing
  `CrossAttention`, `AdaptiveGatedFusion` and `SharedMultimodalEncoder` rather
  than duplicating them.
- **Deployment truth in checkpoints**: checkpoints now record architecture
  name, schema version and runtime metadata so a rebuild is validated by
  construction — and all of it stays `weights_only`-loadable.
- **Existing suites preserved**: the full repo suite passes with the previous
  phase intact (see §8).

## 2. New / extended components

| Module | Type | Responsibility |
| --- | --- | --- |
| `training/models/fusion_engine.py` | new | `CrossModalFusionEngine` + `FusionOutput` — cross-attention + gated fusion + shared encoder as one unit |
| `training/models/runtime.py` | new | precision (AMP), device, `torch.compile`, gradient checkpointing, `DataParallel` / `DDP`, `apply_runtime` |
| `training/models/config.py` | extended | `FusionConfig`, `RuntimeConfig`, `ModelConfig.architecture_version`; template + validator for `use_temporal_stream` |
| `training/models/cropfusion.py` | extended | engine integration, optional temporal stream (`temporal_proj`), `shared_encoder` property, `metadata`, upgraded `summary`, runtime methods |
| `training/models/adaptive_gate.py` | extended | optional fourth temporal stream (`temporal_dim`, `temporal_gate` / `temporal_token`) |
| `training/models/cross_attention.py` | extended | per-call `return_attention` override (backward compatible) |
| `training/models/tabtransformer.py`, `temporal_transformer.py`, `shared_encoder.py` | extended | `set_gradient_checkpointing` + recompute-aware forward |
| `training/models/factory.py` | extended | architecture registry + resolution, `from_checkpoint` rebuild by name, runtime helpers |
| `training/models/checkpoint.py` | extended | stores `architecture`, `architecture_version`, JSON-safe `metadata` |
| `training/models/utils.py` | extended | `architecture_report` (per-module real input/output shapes) |
| `training/models/__init__.py` | extended | exports the new surface |
| `training/models/tests/` | new | `test_fusion_engine.py`, `test_runtime.py`, `test_architecture.py` |

## 3. CrossModalFusionEngine

- Owns the whole cross-modal pathway: `CrossAttention` (Q=image, K=V=tabular) →
  `AdaptiveGatedFusion` (image / tabular / fusion gates, optional temporal) →
  `SharedMultimodalEncoder` (CLS-pooled transformer → `[B, out_dim]`).
- Returns a typed `FusionOutput` (`shared_embedding`, `fused`, `cross_output`,
  projected tokens, per-sample `gates`).
- Built only for multimodal models; single-modality models keep a plain shared
  encoder. The model's `shared_encoder` property hides which path is active.
- Config-driven ablations: `cross_attention.enabled`, `gated_fusion.enabled`,
  `fusion.residual_fusion`, `fusion.use_temporal_stream` — no code changes.
- `return_attention=True` exposes head-averaged cross-attention weights
  (`[B, 1, 1]`) under `gates["cross_attention"]`.
- Temporal stream: when `fusion.use_temporal_stream` is on, the mask-aware mean
  of the fused per-timestep features is projected to the image-embedding width
  and gated as a fourth stream (`temporal_gate`, `temporal_embedding` on the
  model output).

## 4. Runtime support

- `RuntimeConfig` (env `MODEL_` / YAML): `precision`, `device`, `compile` +
  `compile_mode`, `gradient_checkpointing`, `data_parallel`, `distributed`,
  `local_rank`.
- `apply_precision` keeps LayerNorm / BatchNorm in float32; records precision.
- `amp_context` runs blocks under `torch.autocast` (`float32` = no-op);
  unsupported precision/device combos raise `MDL-CONFIG-001` cleanly.
- `compile_model` wraps `torch.compile` (`MissingDependencyError` when
  unavailable); `wrap_data_parallel` / `wrap_distributed` raise
  `ModelConfigurationError` when their prerequisites (CUDA / initialised
  process group) are missing — no silent CPU fallback.
- `apply_runtime` applies everything in a fixed order; `ModelFactory`
  `apply_runtime` / `create_with_runtime` are one-call entry points.
- Gradient checkpointing (`use_reentrant=False`) on the three transformer
  stacks, training-mode-only; the temporal transformer passes its mask through
  as `src_key_padding_mask` so the recompute path matches the normal path.

## 5. Architecture registry, metadata, summary

- `ModelFactory._ARCHITECTURES` registry: `cropfusion_v1` built-in,
  `register_architecture` for future architectures, strict `architecture=`
  construction, config-name fallback for display names, rebuild-by-name in
  `from_checkpoint`.
- `architecture_version` (schema version) on `ModelConfig`; stored in
  checkpoints and `model.metadata`.
- `CheckpointManager.save` now persists `architecture`, `architecture_version`
  and JSON-safe `metadata` alongside config + weights.
- `model.summary(sample_batch=...)` gains `architecture_report` (per-module real
  shapes via `utils.architecture_report`), `input_shapes` / `output_shapes` and
  `metadata`.

## 6. Documentation

- 5 guides: `training/models/docs/FUSION_ENGINE.md`, `RUNTIME.md`,
  `ARCHITECTURES.md`, `SUMMARY.md`, `GRADIENT_CHECKPOINTING.md`.
- 4 diagrams under `docs/diagrams/`: `r2-4-model-architecture.md`,
  `r2-4-cross-modal-fusion.md`, `r2-4-model-runtime.md`,
  `r2-4-model-checkpoint-architecture.md`.

## 7. Verification

- New test modules (42 tests): `test_fusion_engine.py` (12), `test_runtime.py`
  (17), `test_architecture.py` (13).
- `training/models/tests` — **173 passed, 0 failed** (existing 131 + new 42).
- Full repo `pytest` → **1134 passed, 0 failed** (previous 1092 + 42).
- Fixes during verification: `TemporalTransformer` checkpoint path passed the
  key-padding mask positionally (landed on `src_mask`); checkpoint `metadata`
  JSON-serialised so `torch.load(weights_only=True)` accepts `TorchVersion`;
  fp16-weight inference test casts inputs alongside weights (as real AMP use
  does); CPU `float16` autocast support is version-dependent and tested as
  run-or-clean-error.

## 8. Migration impact

| Aspect | Impact |
| --- | --- |
| `training/models/*` | `fusion_engine.py` + `runtime.py` added; all other modules extended additively (no existing test modified) |
| R2.3 packages (STAM, FE, quality, export) | None — untouched |
| Preprocessing / datasets | None — batch contract unchanged |
| Training engine / evaluation / prediction | None — not implemented in this phase |
| `shared/*` | None — `CropFusionError` reused |
| Artifact surface | checkpoints gain `architecture` / `architecture_version` / `metadata` (backward compatible reads) |
