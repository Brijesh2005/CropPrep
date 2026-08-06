# Migration Report — R4 → R5 (Evaluation, Explainability & Inference Package)

Detailed record of the R5 phase: **post-training evaluation** with explainability
extensions, an **ablation study**, **error analysis**, a **model exporter**, an
**inference package generator** with **versioning** and **package validation**,
and six end-of-run reports — the final evaluation/inference surface before the
(separately scheduled) prediction platform. Companion to
[MIGRATION_REPORT_R4](MIGRATION_REPORT_R4.md).

- **Status**: complete
- **Date**: 2026-08-06
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Evaluation and inference packaging only**: this phase implements the
  evaluation suite, explainability extensions, ablation study, error analysis,
  model export and the versioned, validated inference package. It does **not**
  implement the prediction platform / FastAPI / React / Docker / deployment
  surface — those remain out of scope for a later phase.
- **Extend, don't duplicate**: the evaluation suite reuses the Phase-4 loader
  contract and the Phase-5 model surface; the exporter wraps the Phase-5
  exporter (TorchScript / ONNX) and adds the `pytorch` bundle format.
- **Backward compatible**: no existing `training/training` test was modified;
  the model is consumed unchanged (no new model code). New behaviour is
  additive (two new packages plus explainability extensions).
- **Deterministic, tested**: 70 new tests (evaluation 43, inference 27). The
  full repo suite stays green (see §7).
- **Config-first**: all new behaviour is config-driven (`EVAL_*` / `INF_*` env
  > YAML > defaults); env lists use JSON, e.g.
  `INF_EXPORTER__FORMATS='["pytorch","onnx"]'`.

## 2. New / extended components

| Module | Type | Responsibility |
| --- | --- | --- |
| `training/evaluation/` | new | `exceptions.py`, `config.py`, `metrics.py`, `evaluator.py`, `comparison.py`, `ablation.py`, `error_analysis.py`, `reports.py`, `__init__.py` (`__version__ = "0.1.0"`) |
| `training/inference/` | new | `exceptions.py`, `config.py`, `versioning.py`, `exporter.py`, `dataset_sources.py`, `package_builder.py`, `validate.py`, `reports.py`, `__init__.py` (`__version__ = "0.1.0"`) |
| `training/explainability/visualization.py` | extended | `Visualizer.fusion_weights(gates, path)` — per-gate bar chart with 0.5 threshold colouring |
| `training/explainability/facade.py` | extended | writes `fusion_weights.png` when the explanation carries per-sample gates |
| `training/evaluation/tests/` | new | `test_config.py`, `test_metrics.py`, `test_evaluator.py`, `test_comparison.py`, `test_ablation.py`, `test_error_analysis.py`, `test_reports.py` (+ `conftest.py`) |
| `training/inference/tests/` | new | `test_config.py`, `test_versioning.py`, `test_exporter.py`, `test_dataset_sources.py`, `test_package_builder.py`, `test_validate.py`, `test_reports.py` (+ `conftest.py`) |
| `training/evaluation/docs/`, `training/inference/docs/` | new | 5 R5 guides |
| `docs/diagrams/` | new | 4 R5 diagrams (`r5-*.md`) |

## 3. Evaluation

- `MultimodalEvaluator(model, config).evaluate(loader)` runs the Phase-4
  loader in `eval()` / `no_grad` and reduces per-task **extended metrics**
  (classification: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC,
  AUPRC, confusion matrix, per-class; regression: MSE, RMSE, MAE, median
  absolute error, bias, R², MAPE, within-tolerance, error histogram), PR curves,
  confusion matrices, raw predictions, shared embeddings, forward latency and —
  new — per-sample fusion gates into one `EvaluationOutcome`.
- Latency is measured with CUDA/CPU timing (`torch.cuda.Event` when available),
  returning mean / p50 / p95.
- `EvaluationConfig` resolves `EVAL_*` env > YAML (`EVAL_CONFIG_FILE`) >
  defaults; sections `general`, `metrics`, `comparison`, `ablation`,
  `error_analysis`. Lists through env use JSON (e.g. `error_percentiles`).
- `comparison.py`: `compare_models` runs two models (or two configs) over the
  same loader and returns a `MultimodalComparison` (shared metrics) +
  per-model outcomes; `generate_comparison_report` writes the side-by-side
  markdown/JSON.

## 4. Ablation study

- `DEFAULT_VARIANTS` registry of seven variants against one base `ModelConfig`:
  `without_tabtransformer`, `without_efficientnet`, `without_temporal_encoder`,
  `without_cross_attention`, `without_adaptive_gate`,
  `without_confidence_fusion`, `without_temporal_branch`.
- Model-side constraints handled explicitly:
  - no `TemporalModelConfig.enabled` toggle exists → the temporal ablation is
    model surgery: `_TemporalPooling` (mask-aware mean + Linear, same
    input/output width) replaces `temporal_transformer`;
  - no explicit "Confidence Fusion" toggle → mapped to
    `gated_fusion.enabled=False` + `fusion.residual_fusion=False`;
  - `without_efficientnet` also sets `fusion.use_temporal_stream=False` (the
    temporal gate requires the image branch).
- `AblationStudy.run(loader, variants=None)` evaluates each variant, measures
  parameter count and inference ms (with warmup), computes the delta /
  speed-up vs. the full model and picks `best_variant` under
  `compare_metric` / `compare_mode`. `build_variant_config` /
  `apply_variant_surgery` are public.

## 5. Error analysis + explainability tie-in

- `ErrorAnalysis.analyze(outcome, sample_metadata)` produces per-task reports:
  per-class error rates and top confusion pairs, misclassified samples,
  regression bias/outliers/failures, group breakdowns by optional per-sample
  metadata (village / district / season / year) and — new — a
  `fusion_analysis` of mean gate values over overall / correct / error buckets.
- The evaluator collects per-sample gates (`EvaluationOutcome.gates`); the
  error-analysis figure `fusion_gates.png` plots `image_gate` / `tabular_gate`
  / `fusion_gate` means for correct vs. failing samples — the quantitative
  answer to "does the model lean on the wrong modality when it fails?".
- `Visualizer.fusion_weights` in explainability renders the same gate values as
  a horizontal bar chart and is emitted by `facade.py` when gates are present.
- `sample_metadata` length mismatch raises `ErrorAnalysisError`.

## 6. Model export + inference package

- `ModelExporter.export_bundle` wraps the Phase-5 exporter (TorchScript trace +
  ONNX, opset 17, dynamic batch) and adds the **pytorch** format: a
  self-describing `cropfusion.pt` payload (`format`, `format_version`,
  `config`, `state_dict`, `metadata`) so a consumer can rebuild the model via
  `ModelFactory` without the training package. `load_pytorch_model` rebuilds
  and returns model + metadata. Sidecars: `model_config.yaml`, `metrics.json`,
  `metadata.json`, `checksums.json`.
- `PackageBuilder.build` assembles the versioned package directory
  `cropfusion-<version>/` with 14 required artifacts (one per bundle format),
  the manifest, checksums, runtime-side modules, `requirements.txt`, `api.py`,
  `inference_adapter.py`, `README.md` and reproducibility metadata.
- `versioning.py`: semver triple from explicit `major/minor/patch` or
  `next_version` disambiguation; auto-numbering from existing package dirs;
  fingerprint-conflict detection raises `VersionConflictError`.
- `validate.py`: `validate_integrity` (SHA-256 vs `checksums.json`),
  `validate_manifest`, `validate_compatibility` (fingerprint + parameter
  count) and `smoke_test` (builds a batch from the model's own config — no
  external `sample_batch`). `package_validator` wires the four checks;
  failures raise `PackageValidationError`.
- Known edge cases handled: `checksums.json` / `manifest.json` excluded from
  their own checksum map; manifest `formats` resolved against
  `BUNDLE_FORMAT_FILES`; the validator skips `checksums.json` in the
  required-artifact check; `_training_config` stored on the builder so the
  training fingerprint flows through; `BuildReport.to_dict()` / report markdown
  render `ValidationResult` objects correctly.
- **Six reports**: `evaluation_report`, `comparison_report`, `ablation_report`,
  `error_analysis_report`, `export_report`, `inference_package_report` (+ the
  embedded `validation_report`). A full E2E build produced 18 artifacts,
  including optional `cropfusion.onnx`, with all four validation checks green
  and manifest formats `["pytorch","onnx"]`.

## 7. Verification

- New tests: `training/evaluation/tests` — **43 passed**;
  `training/inference/tests` — **27 passed** (total **70 new**).
- Full repo `pytest` → **1256 passed, 0 failed** (previous 1186 + 70).
- `python -m compileall -q` clean for `training/evaluation`,
  `training/inference` and the extended explainability modules.
- E2E inference smoke passed (18 artifacts, all checks green) as described in §6.

## 8. Migration impact

| Aspect | Impact |
| --- | --- |
| `training/evaluation/*`, `training/inference/*` | new packages; no existing code touched |
| Phase 5 models / R2.4 runtime / R4 trainer | None — consumed unchanged |
| `training/explainability/*` | `visualization.py` + `facade.py` extended additively (gate figures) |
| R2.3 packages (STAM, FE, quality, export) | None — untouched |
| `shared/*` | None — `CropFusionError` / error codes reused per package (`EvaluationError`/`InferenceError` families) |
| Prediction platform / FastAPI / React / Docker | None — explicitly out of scope for R5 |
| Artifact surface | `cropfusion.pt|.torchscript.pt|.onnx`, model config, schema, metrics, metadata, checksums, manifest, validation reports |
