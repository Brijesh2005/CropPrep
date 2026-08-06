# Migration Report — R3 → R4 (Training Framework)

Detailed record of the R4 phase: the **training framework** that completes the
Phase 7 strategy surface for the Phase 5 `CropFusionModel` — a dedicated
`CropFusionTrainer` with five-stage curriculum training, class-frequency
weighted losses, `torch.compile` wiring and five end-of-run reports.
Companion to [MIGRATION_REPORT_R2.4](MIGRATION_REPORT_R2.4.md).

- **Status**: complete
- **Date**: 2026-08-06
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Training only**: this phase implements training orchestration on top of the
  existing engine — it does **not** implement the prediction platform / FastAPI
  / React / Docker / inference / deployment surface. Those remain out of scope.
- **Extend, don't duplicate**: the R4 trainer extends the existing
  `training/training` package rather than creating a second trainer — the base
  `Trainer`, losses, optimizers, schedulers, validators, checkpoints and
  callbacks are reused, so there is exactly one training engine.
- **Backward compatible**: no existing `training/training` test was modified;
  every existing public API still works. New behaviour is additive (new config
  sections, new modules, one new trainer subclass).
- **Deterministic, tested**: 52 new tests cover scheduling, freezing, resume,
  weight recipes, the full trainer flow and every report. The full repo suite
  stays green (see §7).
- **Config-first**: all new behaviour is config-driven (`TRN_*` env > YAML >
  defaults) and defaults to off unless the brief requires it on (reports on,
  curriculum off, class weighting off, compile off).

## 2. New / extended components

| Module | Type | Responsibility |
| --- | --- | --- |
| `training/training/curriculum.py` | new | `CurriculumStage`, `CURRICULUM_STAGES`, `STAGE_ORDER`, `stage_for`, `Curriculum`, `build_curriculum`, `CurriculumCallback` |
| `training/training/cropfusion_trainer.py` | new | `CropFusionTrainer`, `CropFusionTrainingResult`, class-frequency collection, curriculum + report wiring |
| `training/training/reports.py` | new | `generate_reports` + 5 report generators, `default_reports_dir`, `REPORT_TYPES` |
| `training/training/losses.py` | extended | `WeightedLabelSmoothingLoss`, `compute_class_counts`, `class_frequency_weights`, `build_class_weights`, weighted `build_task_loss` / `MultiTaskLoss` |
| `training/training/config.py` | extended | `CurriculumConfig`; `LossConfig.class_weight_*`; `GeneralConfig.compile*` / `reports` / `reports_dir` |
| `training/training/trainer.py` | extended | `raw_model` + optional `torch.compile`; `_on_model_train_mode` hook for per-module mode splits |
| `training/training/__init__.py` | extended | exports the new surface |
| `training/training/tests/` | new | `test_curriculum.py`, `test_class_weights.py`, `test_cropfusion_trainer.py`, `test_reports.py`; `conftest.py` extended (`multimodal` fake loader) |

## 3. Curriculum training

- Five stages: `tabular` → `image` → `temporal` → `fusion` → `finetune`,
  keyed to top-level `CropFusionModel` components (`tab_encoder`; `ndvi_encoder`
  / `evi_encoder` / `image_fusion`; `temporal_transformer` / `temporal_proj`;
  `fusion_engine`; everything).
- `Curriculum.active_stages()` drops stages whose components do not exist on
  the model (a tabular-only model runs `tabular → finetune`) and their epoch
  budget merges into the survivors.
- Scheduling: `stage_epochs()` splits the epoch budget across the active stages
  — `epochs_per_stage` overrides individual stages, the remainder is split
  evenly. `start_stage` (1..5) skips earlier stages (resume-from-any-stage).
- Freeze semantics: frozen parameters get `requires_grad=False` **and** their
  modules run in `eval()` mode (BatchNorm statistics / Dropout inert); the
  trainable scope stays in `train()` mode.
- `CurriculumCallback` applies the stage on `on_epoch_begin` on **every rank**
  (`all_ranks=True`, DDP consistency) and re-applies the eval-mode split after
  each `model.train()` via the trainer's `_on_model_train_mode` hook.
- The optimizer is built after the first stage is applied, so only trainable
  parameters receive optimizer slots.

## 4. Class-frequency weighted losses

- One pass over the training loader collects crop-label counts
  (`_collect_class_counts`); `build_class_weights` turns them into a `[C]`
  tensor for `balanced` (`N/(C·n_c)`), `sqrt_inv` (`1/√n_c`) or `effective_num`
  (Cui et al., 2019). Weights are floored at `class_weight_eps`, normalised to
  mean 1 and stable for zero-count classes. No oversampling; focal loss remains
  future work.
- Weights thread into the crop loss only: `CrossEntropyLoss(weight=)`,
  `WeightedLabelSmoothingLoss(weight=)`, `FocalLoss(alpha=)`. The yield
  regression task is never weighted.
- `WeightedLabelSmoothingLoss.weight` is a registered buffer (device-safe,
  moves with the loss module); `MultiTaskLoss` accepts
  `class_weights={"crop": ...}`. A user-supplied `loss_module` is respected
  unchanged.

## 5. CropFusionTrainer, compile wiring, reports

- `CropFusionTrainer(Trainer)`:
  - computes class-frequency weights and injects them into the multi-task loss;
  - builds the curriculum and prepends its callback (before user callbacks);
  - records the active `stage` in epoch logs (`log_transitions`);
  - returns a `CropFusionTrainingResult` (`stages`, `reports`,
    `summary()` extensions) and writes the reports at the end of `train()`.
  - Every other behaviour (AMP, gradients, schedulers, checkpoints, callbacks,
    resume) is inherited.
- `torch.compile` wiring: `Trainer` keeps `raw_model`, optionally wraps it with
  `torch.compile(mode=..., backend=...)` before the DDP wrapper; missing
  `torch.compile` support raises `MissingDependencyError` cleanly.
  `compile_backend: "eager"` is a cheap wiring-exercise path used by tests.
- Reports (`general.reports`, default on) land in `<output_dir>/reports` unless
  `reports_dir` is set: `training_report.md`, `validation_report.md`,
  `metrics_report.md`, `checkpoint_report.md`, `learning_curve.csv` (scalars
  only). All values come from in-memory history.

## 6. Documentation

- 5 guides: `training/training/docs/CURRICULUM.md`, `CLASS_WEIGHTS.md`,
  `CROPFUSION_TRAINER.md`, `COMPILE.md`, `REPORTS.md`.
- 5 diagrams under `docs/diagrams/`: `r4-training-pipeline.md`,
  `r4-curriculum-flow.md`, `r4-optimizer-flow.md`, `r4-checkpoint-flow.md`,
  `r4-validation-flow.md`.

## 7. Verification

- New test modules (52 tests): `test_curriculum.py` (19), `test_class_weights.py`
  (16), `test_cropfusion_trainer.py` (8), `test_reports.py` (9).
- `training/training/tests` — **119 passed, 0 failed** (existing 67 + new 52).
- Full repo `pytest` → **1186 passed, 0 failed** (previous 1134 + 52).
- Fixes during verification:
  - `WeightedLabelSmoothingLoss` with `weight=None` left no attribute (earlier
    a plain `self.weight = None` collided with `register_buffer`); the buffer
    is now always registered.
  - `CurriculumCallback.stages_log` used a `Field` default on a non-dataclass
    (→ `'Field' object has no attribute 'append'`); plain list initialised in
    `__init__`.
  - `learning_curve_csv` passed a list to `csv.writer`; now uses `io.StringIO`.
  - Tabular-only model + curriculum raised `element 0 of tensors does not
    require grad` because image/temporal/fusion stages reference missing
    components ⇒ everything frozen; `active_stages()` now filters them out.
  - Test-side: full-model tests needed `feature_dim=4` (`numeric_dim +
    len(cardinalities)`) and a `multimodal` fake loader (`ndvi`/`evi`
    `[B,T,1,H,W]` + `temporal_mask`); a weighted-smoothing expectation had to
    average over all samples; one full-flow test seeded the loader RNG and
    raised early-stopping patience so the val-loss trajectory could not
    truncate the run.

## 8. Migration impact

| Aspect | Impact |
| --- | --- |
| `training/training/*` | `curriculum.py`, `cropfusion_trainer.py`, `reports.py` added; `trainer.py`, `losses.py`, `config.py`, `__init__.py` extended additively (no existing test modified) |
| Phase 5 models / R2.4 runtime | None — the trainer consumes the existing model API |
| R2.3 packages (STAM, FE, quality, export) | None — untouched |
| `shared/*` | None — `CropFusionError` / `TR-*` codes reused |
| Prediction platform / FastAPI / React / Docker | None — explicitly out of scope for R4 |
| Artifact surface | `training_report.md`, `validation_report.md`, `metrics_report.md`, `checkpoint_report.md`, `learning_curve.csv`; checkpoints unchanged |
