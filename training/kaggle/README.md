# Kaggle experiments — Training Infrastructure (R2.1)

CropFusion model exploration runs on Kaggle GPUs. This directory holds the
Kaggle Training Infrastructure: orchestration notebooks, orchestration scripts,
the pinned Kaggle runtime and the configuration layer. All heavy engines live in
the Training Platform packages (`training.dataset_manager`, `training.stam`,
`training.preprocessing`, `training.training`, `training.models`); the scripts
and notebooks only wire configs + data + entry points.

**Phase scope (R2.1):** infrastructure only — environment detection, logging,
workspace, checkpoint metadata, cache, validation and reports. No training,
evaluation or export logic is implemented here yet.

## Layout

```
training/kaggle/
├── notebooks/            train / evaluate / export / system_check (R2.1 infra)
├── scripts/              bootstrap.py, run_training.py, system_check.py (+ evaluate, export_release)
├── config.py             validated platform config (KAGGLE_* env + extends chains)
├── environment/          Environment Manager (runtime / system / GPU / dependencies)
├── logging.py            Training Logger (startup / system / experiment logs)
├── workspace.py          Workspace Manager (folders, cache clean, resume, outputs)
├── checkpoints.py        Checkpoint Manager (metadata, latest/best/resume, versioning)
├── cache.py              Training Cache (5 JSON buckets + TTL/LRU)
├── validation.py         Training Validator (config/python/GPU/deps/folders/permissions/disk/providers)
├── reports.py            environment/gpu/dependency/storage/workspace/configuration reports
├── docs/                 SETUP, KAGGLE, BOOTSTRAP, WORKSPACE, CONFIGURATION guides
└── configs/ logs/ outputs/ checkpoints/ cache/   (runtime dirs, git-ignored)
```

## Typical run (on Kaggle)

1. Attach the crop-yield dataset to the notebook (or use `bootstrap.py --ensure-data`).
2. Run `notebooks/system_check.ipynb` → validation report + all reports.
3. Run `notebooks/train.ipynb` → bootstrap + pipeline orchestration readiness.
4. Run `notebooks/evaluate.ipynb` → checkpoint registry status.
5. Run `notebooks/export.ipynb` → exporter/release wiring check.

## Research machine

```bash
export PYTHONPATH="$PWD"
python training/kaggle/scripts/bootstrap.py --repo-root . --skip-install
python training/kaggle/scripts/run_training.py --repo-root .
python training/kaggle/scripts/system_check.py --repo-root .
```

## Config

Configs are shared with local runs (`training/config/{dataset,training,model,
validation,kaggle,logging,paths}.yaml`), so a notebook behaves exactly like a
local run. See [Configuration Guide](docs/CONFIGURATION.md).

## Docs

- [Training Setup Guide](docs/SETUP.md)
- [Kaggle Guide](docs/KAGGLE.md)
- [Bootstrap Guide](docs/BOOTSTRAP.md)
- [Workspace Guide](docs/WORKSPACE.md)
- [Configuration Guide](docs/CONFIGURATION.md)
