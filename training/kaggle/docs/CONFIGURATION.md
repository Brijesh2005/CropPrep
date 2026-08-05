# Configuration Guide

The Training Platform loads three configuration documents through
`training/kaggle/config.py`, plus a registry pointing at every config file.

## Precedence

1. Environment variables — `KAGGLE_<SECTION>__<KEY>` (nested via `__`).
2. YAML configuration files.
3. Pydantic defaults.

Example env override:

```bash
export KAGGLE_WORKSPACE__ROOT="training/kaggle"       # workspace.root
export KAGGLE_KAGGLE__GPU="false"                     # kaggle.gpu
```

## Documents

### `training/config/paths.yaml` — workspace + registry + requirements

```yaml
extends: null                       # optional parent paths file (chainable)
workspace:                          # WorkspaceLayout source
  root: training/kaggle
  logs_dir: logs
  outputs_dir: outputs
  checkpoints_dir: checkpoints
  cache_dir: cache
  configs_dir: configs
config:                             # registry of every config file
  dataset: training/config/dataset.yaml
  training: training/config/training.yaml
  kaggle: training/config/kaggle.yaml
  model: training/config/model.yaml
  logging: training/config/logging.yaml
  paths: training/config/paths.yaml
  validation: training/config/validation.yaml
environment:                        # Training Validator requirements
  min_python: "3.10"
  require_gpu: true
  min_free_gb: 5.0
  required_dependencies: [numpy, pandas, torch, scikit-learn, rasterio, yaml, kagglehub]
  gpu_dependencies: [torch]
```

**Inheritance:** `extends` names a parent paths file resolved relative to the
child; the parent is deep-merged under the child (child wins), then env
overrides win. Chains are supported.

### `training/config/kaggle.yaml` — Kaggle runtime

`kaggle:` (dataset handle, competition, gpu, internet), `runtime:` (repo/input/
working dirs, artifacts root, config dir), `dataset_manager:` (config file +
roots), `outputs:` (run/checkpoint/export/release dirs), `install:`
(editable packages).

### `training/config/logging.yaml` — Training Logger

`level`, `dir`, `max_bytes`, `backup_count`, `json_format`, `console`. Mirrors
the Dataset Manager `LogConfig` so it can be merged into `dataset.yaml`.

## Registry (config section of paths.yaml)

The Training Validator checks every registry file exists and parses as a YAML
mapping:

- `dataset.yaml` — Dataset Manager (`DM_*` overrides).
- `training.yaml` — training engine (`TRN_*` overrides).
- `model.yaml` — model config (`MOD_*` overrides).
- `kaggle.yaml` — Kaggle runtime.
- `logging.yaml` — logging.
- `paths.yaml` — this file.
- `validation.yaml` — STAM validation.

## Loaders

```python
from training.kaggle.config import (
    load_paths_config, load_kaggle_config, load_logging_config,
    WorkspaceLayout,
)
paths  = load_paths_config("training/config/paths.yaml")
layout = WorkspaceLayout.resolve(paths)          # repo-rooted Path objects
```

`WorkspaceLayout` is a frozen dataclass with `.root/.logs/.outputs/.
checkpoints/.cache/.configs`; any directory can be overridden for testing, e.g.
`WorkspaceLayout.resolve(paths, logs=Path("/tmp/logs"))`.
