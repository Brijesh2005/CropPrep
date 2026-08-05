# Kaggle experiments

CropFusion model exploration runs on Kaggle GPUs. This directory holds the
orchestration notebooks, the orchestration scripts and the pinned Kaggle
runtime. All heavy engines live in the Training Platform packages
(`training.dataset_manager`, `training.stam`, `training.preprocessing`,
`training.training`, `training.models`); the scripts and notebooks only wire
configs + data + entry points.

## Layout

- `notebooks/train.ipynb` — bootstrap environment + run an end-to-end training run.
- `notebooks/evaluate.ipynb` — reload a checkpoint and evaluate on a hold-out test set.
- `notebooks/export.ipynb` — export TorchScript / ONNX + `release.json` manifest.
- `scripts/bootstrap.py` — install editable packages, add repo root, report provider manifests (`--ensure-data` to download the Kaggle imagery).
- `scripts/run_training.py` — Dataset Manager ensure → STAM observations → `run_experiment`.
- `scripts/evaluate.py` — `ModelFactory.from_checkpoint` → `Evaluator` over the hold-out test set.
- `scripts/export_release.py` — `ModelExporter` TorchScript/ONNX + model config + release manifest.
- `requirements.txt` — minimal Kaggle runtime dependencies.
- `setup.py` - placeholder packaging so notebooks can `pip install -e ./training/kaggle`.

Configs are shared with local runs (`training/config/{dataset,training,model,
validation,kaggle}.yaml`), so a notebook behaves exactly like a local run.

## Typical run (on Kaggle)

1. Attach the crop-yield dataset to the notebook (or use `bootstrap.py --ensure-data`).
2. Run `notebooks/train.ipynb` → report JSON under `runs/<run-name>`.
3. Run `notebooks/evaluate.ipynb` → `evaluation.json` (metrics + latency/memory artifacts).
4. Run `notebooks/export.ipynb` → TorchScript + ONNX + `release.json`.

## Research machine

```bash
python training/kaggle/scripts/bootstrap.py --repo-root . --skip-install
python training/kaggle/scripts/run_training.py --repo-root . \
    --locations training/kaggle/locations.csv
```
