# Kaggle experiments

CropFusion model exploration runs on Kaggle GPUs. This directory holds the
notebooks, helper scripts and the pinned Kaggle runtime.

## Layout

- `notebooks/` - Kaggle notebooks (data prep, baseline models, ablation runs).
- `scripts/` - small helper scripts shared across notebooks.
- `requirements.txt` - minimal Kaggle runtime dependencies.
- `setup.py` - placeholder packaging so notebooks can `pip install -e ./training/kaggle`.

Notebooks import the same first-party packages as local runs
(`training.models`, `training.preprocessing`, ...) - see the root README for
installation.
