# Bootstrap Guide

`training/kaggle/scripts/bootstrap.py` prepares a Kaggle notebook or GPU
research machine to run the Training Platform. It is **orchestration only** —
no training or dataset implementation.

## Responsibilities

1. **Verify the Kaggle runtime** (is this a notebook? where are input/working dirs?).
2. **Verify Python version** (>= `environment.min_python` in `paths.yaml`).
3. **Verify CUDA / GPU** (torch + `nvidia-smi`).
4. **Install dependencies** — `pip install -e` the editable packages listed in
   `training/config/kaggle.yaml` (`install.editable_packages`).
5. **Initialise configuration** — loads + validates `paths.yaml`, `kaggle.yaml`
   and `logging.yaml` with `KAGGLE_*` environment overrides.
6. **Initialise logging** — the Training Logger (startup/system/experiment logs).
7. **Initialise workspace** — creates `logs/ outputs/ checkpoints/ cache/ configs/`.
8. **Initialise Dataset Manager providers** — reports the tabular + image
   provider manifests.
9. **Verify repository integrity** — git repo present, `training/` and
   `training/config/` exist.
10. **Verify tabular datasets** — counts the tabular CSVs.
11. **Verify the image dataset** — provider availability (+ materialise with
    `--ensure-data`).
12. **Generate the startup + environment reports** under
    `training/kaggle/outputs/reports/`.

## CLI

```
--kaggle-config    training/config/kaggle.yaml (default)
--paths-config     training/config/paths.yaml  (default)
--logging-config   training/config/logging.yaml(default)
--dataset-config   training/config/dataset.yaml(default)
--repo-root        repository root (auto-detected)
--skip-install     skip the editable pip installs
--ensure-data      materialise the Kaggle imagery dataset (download-or-reuse)
--output           report directory (default: workspace outputs/reports)
```

## Exit / readiness

The final line prints `tabular=... image=... repo_integrity=... -> READY/NOT READY`.
READY requires the tabular Git CSVs to be available and repository integrity to
hold. Imagery can be attached as a Kaggle dataset or materialised with
`--ensure-data`.

## Reports written

`environment.json`, `gpu.json`, `dependency.json`, `storage.json`,
`workspace.json`, `configuration.json`, `bootstrap.json`.

## Related

- [Training Setup Guide](SETUP.md) — local setup + bootstrap.
- [Workspace Guide](WORKSPACE.md) — what bootstrap creates.
- [Configuration Guide](CONFIGURATION.md) — the config files bootstrap loads.
