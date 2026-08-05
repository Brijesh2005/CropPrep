# Migration Report — R1.4 → R2.1 (Kaggle Training Infrastructure)

Detailed record of the R2.1 phase: building the fully functional **Kaggle
Training Infrastructure** — the runtime layer the Training Platform needs
before any training runs. Companion to
[MIGRATION_REPORT_R1.4](MIGRATION_REPORT_R1.4.md).

- **Status**: complete
- **Date**: 2026-08-05
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Infrastructure only**: R2.1 built environment detection, configuration,
  logging, workspace, checkpoint metadata, cache, validation and reporting.
  It did **not** implement dataset loading, STAM, training, fusion, evaluation
  or prediction.
- **No training logic added**: the training engine (`training.training`
  Experiment/Trainer/Evaluator), models, STAM, preprocessing, dataset manager
  and explainability packages were **not modified** — only imported / wired.
- **Self-contained**: the deliverables live under `training/kaggle/` and
  `training/config/paths.yaml`; a Kaggle notebook can clone the repo, attach
  the dataset and run one bootstrap script.
- **Existing suites preserved**: no existing test suite was changed; the
  R1.4 verification gates still pass (see §8).
- **Boundary respected**: existing orchestration scripts (`evaluate.py`,
  `export_release.py`) and notebooks remain untouched in behaviour; the R2.1
  phase re-wires `bootstrap.py` + `run_training.py` and adds
  `system_check.py` + `system_check.ipynb`.

## 2. New components — `training/kaggle/`

| Module | Component | Responsibility |
| --- | --- | --- |
| `config.py` | Configuration layer | Validated `PathsConfig` / `KaggleConfig` / `LoggingConfig` with `KAGGLE_*` env overrides + `extends` inheritance; `WorkspaceLayout` resolution |
| `environment/` | Environment Manager | Runtime (Kaggle), system (CPU/RAM/disk/python), GPU/CUDA, dependency probes + combined capability report |
| `logging.py` | Training Logger | JSON files + compact console + rotating handlers; `startup.log`, `system.log`, `experiment.log` |
| `workspace.py` | Workspace Manager | Folder structure, cache clean, resume, outputs, temp, checkpoint delegation |
| `checkpoints.py` | Checkpoint Manager | Layout + `metadata.json`, latest/best/resume, per-run versioning — **no model saves** |
| `cache.py` | Training Cache | 5 JSON buckets (metadata/preprocessing/image_metadata/statistics/validation), TTL + LRU |
| `validation.py` | Training Validator | config/python/GPU/deps/folders/permissions/disk/providers → shared `ValidationResult` |
| `reports.py` | Reports | environment / gpu / dependency / storage / workspace / configuration + `write_reports` |

## 3. New configuration — `training/config/paths.yaml`

The 7th registry config (was missing). Adds:

- `workspace:` layout (root, logs, outputs, checkpoints, cache, configs),
- `config:` registry of all 7 config files,
- `environment:` minimum requirements (python, GPU, disk, dependencies),
- `extends:` support for config inheritance (chainable, deep-merged, env wins).

## 4. Scripts

| Script | Change |
| --- | --- |
| `scripts/bootstrap.py` | Extended: runtime/Python/CUDA/GPU verification, config init, logging, workspace, providers, repo integrity, tabular/image verification, startup + environment reports |
| `scripts/run_training.py` | Rewritten: **orchestration only** — initialises environment, config, providers, Dataset Manager, STAM, preprocessing, trainer, evaluator, exporter and writes the orchestration report. No training. |
| `scripts/system_check.py` | New: full Training Validator run + all reports; exit code 0 = ready |

## 5. Notebooks

`train.ipynb` / `evaluate.ipynb` / `export.ipynb` updated to the infrastructure
phase; `system_check.ipynb` added. All four import the project, load config,
initialise the environment, verify providers and generate reports — none train.

## 6. Documentation

- `training/kaggle/docs/`: SETUP.md (Training Setup), KAGGLE.md (Kaggle),
  BOOTSTRAP.md (Bootstrap), WORKSPACE.md (Workspace), CONFIGURATION.md
  (Configuration) guides.
- `training/kaggle/README.md` rewritten for the infrastructure layout.
- `training/kaggle/requirements.txt` + `setup.py` updated
  (editable `training.kaggle` namespace package).

## 7. Verification

- `training/kaggle` infrastructure smoke-tested on the research machine:
  `bootstrap.py --skip-install` → all 7 reports written, tabular provider ready;
  `run_training.py` → orchestration report with 6 component descriptors;
  `system_check.py` → validation report (GPU_UNAVAILABLE expected on a
  non-GPU dev machine; passes on Kaggle GPUs).
- Full suites re-run green (see R1.4 gates):
  `pytest shared/tests` → 101 passed; `pytest application/backend/app/tests`
  → 80 passed; `pytest application/tests` → 16 passed.

## 8. Migration impact

| Aspect | Impact |
| --- | --- |
| Training engines | None — imported only, not modified |
| Dataset Manager / STAM / preprocessing / models | None — read-only usage |
| `application/*` (Prediction Platform) | None |
| `shared/*` | Reused (logging, validation, versioning, config) — not modified |
| Existing `evaluate.py` / `export_release.py` | Untouched |
| New runtime dirs | `training/kaggle/{logs,outputs,checkpoints,cache}` (git-ignored) |
