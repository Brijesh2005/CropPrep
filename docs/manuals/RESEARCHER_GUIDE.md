# Researcher Guide

This guide is for researchers using CropFusion to reproduce results, extend the
model, or run experiments.

## Reproducing the environment

```bash
conda env create -f environment.yml
conda activate cropfusion
pip install -e ./ai/models ./ai/preprocessing ./ai/training ./ai/explainability
pip install -e ./services/dataset_manager ./services/spatial_alignment
```

Pinned versions: `requirements.txt`. First-party packages are installed from
source so you always run the working tree.

## Data pipeline

1. **Datasets** - source CSVs in `Tabular_Datasets/`; managed by
   `services/dataset_manager` (profiling/validation/caching/export). Statistics:
   `research/dataset_stats.json` (regenerate with
   `python research/scripts/dataset_stats.py`).
2. **Alignment** - `services/spatial_alignment` (STAM) joins geography with
   time series.
3. **Preprocessing** - `ai/preprocessing` builds multimodal batches
   (tabular + NDVI/EVI sequences + temporal masks).

## Training and evaluation

The training loop and evaluator live in `ai/training`. See
[research/TRAINING_AND_EVAL.md](../research/TRAINING_AND_EVAL.md) for metrics
(accuracy/precision/recall/F1, RMSE/MAE/R²/MAPE, multi-task score) and the
benchmark methodology in [research/BENCHMARKS.md](../research/BENCHMARKS.md).

```bash
# Record an experiment run
cropfusion-mlops experiment mymodel --accuracy 0.87 --loss 0.31 \
    --dataset-version 2024.1

# Register the checkpoint
cropfusion-mlops register mymodel 0.1.0 checkpoint.pt --accuracy 0.87

# Compare runs
cropfusion-mlops experiments-list --model mymodel
```

Experiments are append-only in `experiments/runs.jsonl` - useful as a
reproducible audit trail.

## Model architecture

- Full architecture: [research/MODEL_ARCHITECTURE.md](../research/MODEL_ARCHITECTURE.md)
  and `ai/models/docs/ARCHITECTURE.md`.
- TabTransformer + Dual CNN (NDVI/EVI, EfficientNetV2-S) + temporal Transformer
  + cross-modal attention + adaptive gated fusion.
- Modality routing allows tabular-only or image-only variants for ablation.

## Quality evaluation

The `quality` package provides:

- **Drift** (`quality/drift`) - feature/label/prediction/spatial/temporal drift.
- **Fairness** (`quality/fairness`) - group metrics + regional fairness.
- **Monitoring** (`quality/monitoring`) - Prometheus exporters + dashboards.
- **Optimization** (`quality/optimization`) - ONNX/TorchScript/quantization
  benchmarking.

```bash
pytest quality -q
python -m mlops.scheduler --once     # run the drift/fairness cycle
```

## Publication support

- Cite the project: [CITATION.md](../CITATION.md).
- Design document: `docs/SOFTWARE_DESIGN_DOCUMENT.md`.
- Phase completion reports document methodology and test counts.
