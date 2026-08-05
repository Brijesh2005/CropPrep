# Training Setup Guide

How to prepare a research machine (or Kaggle GPU notebook) to run the CropFusion
Training Platform. This guide covers the **infrastructure phase (R2.1)** — it
gets the environment, workspace and pipeline wiring ready. Model training,
evaluation and export arrive in later phases.

## Requirements

- Python **>= 3.10** (recommended 3.12).
- CUDA GPU recommended for training (`training/config/paths.yaml` →
  `environment.require_gpu: true`).
- The dependencies listed in `training/kaggle/requirements.txt`.

## 1. Clone + install

```bash
git clone <repo-url> CropPrep
cd CropPrep
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate on Linux
pip install -r training/kaggle/requirements.txt
```

The repo root must be on `sys.path` so the `training.*` packages resolve:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "D:\CropPrep"
# Linux / Kaggle
export PYTHONPATH="$PWD"
```

## 2. Bootstrap

```bash
python training/kaggle/scripts/bootstrap.py --repo-root . --skip-install
```

Bootstrap verifies the runtime, Python version, CUDA/GPU, loads the
configuration, initialises the Training Logger + workspace, verifies the
Dataset Manager providers and writes the reports under
`training/kaggle/outputs/reports/`.

> On a fresh clone run without `--skip-install` so the first-party editable
> packages (`training/models`, `training/preprocessing`, `training/training`,
> `training/dataset_manager`, `training/stam`, `training/explainability`) are
> installed.

## 3. Verify

```bash
python training/kaggle/scripts/system_check.py --repo-root .
```

Exit code `0` means the infrastructure is ready (config, python, GPU,
dependencies, folders, permissions, disk, providers).

## 4. Locate the outputs

| Artefact | Location |
| --- | --- |
| Logs | `training/kaggle/logs/` (`startup.log`, `system.log`, `experiment.log`, `training.log`) |
| Reports | `training/kaggle/outputs/reports/*.json` |
| Checkpoint metadata | `training/kaggle/checkpoints/metadata.json` |
| Training cache | `training/kaggle/cache/<bucket>.json` |
| Resolved configs | `training/kaggle/configs/` |

## Running on Kaggle

Follow the [Kaggle Guide](KAGGLE.md) instead — it attaches the dataset, mounts
the repo into `/kaggle/working/CropPrep` and runs the notebooks.

See also: [Bootstrap Guide](BOOTSTRAP.md), [Workspace Guide](WORKSPACE.md),
[Configuration Guide](CONFIGURATION.md).
