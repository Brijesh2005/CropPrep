# Kaggle Guide

How to run the CropFusion Training Platform on a Kaggle GPU notebook
(**infrastructure phase R2.1** — no training yet).

## Create the notebook

1. Create a notebook with **GPU accelerator** (T4/P100) and **Internet** on.
2. Attach the crop-yield dataset
   `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`.
3. Clone the repository into the working dir:
   `!git clone <repo-url> /kaggle/working/CropPrep`
   (or mount it through a Kaggle dataset).

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/system_check.ipynb` | Validate environment + workspace; write all reports. |
| `notebooks/train.ipynb` | Bootstrap + pipeline orchestration readiness (no training). |
| `notebooks/evaluate.ipynb` | Bootstrap + checkpoint registry status (latest/best/resume). |
| `notebooks/export.ipynb` | Bootstrap + exporter/release wiring check. |

Every notebook starts from `REPO_ROOT = /kaggle/working/CropPrep` and adds it to
`sys.path`, then shells out to `training/kaggle/scripts/*.py`.

## Runtime paths

On Kaggle the runtime paths in `training/config/kaggle.yaml` resolve to:

- repo root → `/kaggle/working/CropPrep`
- input dir → `/kaggle/input` (attached dataset + `--ensure-data` download)
- working dir → `/kaggle/working`

The workspace (`training/config/paths.yaml`) lives inside the repo mount:
`training/kaggle/{logs,outputs,checkpoints,cache,configs}`. Kaggle `/kaggle/working`
is ephemeral — persist reports under `outputs/` and download them before the
session ends.

## Runtime detection

The Environment Manager detects the Kaggle runtime via the `/kaggle` directory
and `KAGGLE_*` environment variables; see
`training/kaggle/environment/runtime.py`. Reports include `is_kaggle`, input /
working dirs, kernel run type and internet status.

## Requirements

Install once per session (or let `bootstrap.py` handle the editable packages):

```bash
pip install -r training/kaggle/requirements.txt
```

> `tensorflow`, `opencv-python` and `GDAL` are **optional** — the Environment
> Manager probes and reports them but never requires them for the
> infrastructure phase.
