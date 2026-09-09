# Phase 4 — Kaggle execution guide (image branch)

The 93 GB Sentinel-2 benchmark **must** run on a Kaggle GPU. All local work is
already done (code + local smoke tests in `training/`). This folder only
contains the thin Kaggle entry point.

## One-time prereq (done)
Kaggle credentials work locally; the dataset is public:

    kaggle datasets list -s crop-yield-forecasting-karnataka-dakshina-kannada

Real dataset handle (note the trailing `n`, unlike earlier drafts):
`shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`.
Metadata exported to `dataset-metadata.json` at repo root (version 10,
datasetId 10915019, ~93.4 GB site listing; description mentions Sentinel-2
indices NDVI / EVI / GCI / Moisture Index, weather & soil CSVs, field records,
2021–2024). The exact on-disk layout is NOT assumed by the code: raster
discovery classifies each file from its filename (index token + date) and the
sample↔file matching is enforced by geodata (lat/lon, year, season windows).

## Create the notebook
1. kaggle.com/new/notebook → choose the **GPU** accelerator
   (T4 integer is enough; P100 also fine).
2. **+ Add input** → search `crop-yield-forecasting-karnataka-dakshina-kannada`
   (owner **Shathananda Bhat N**) → attach.
3. It must be able to run the code: either
   - **git clone** the repo (use a private-repo access token set as the
     notebook env var `GITHUB_TOKEN`), then prefix commands with `!`, or
   - upload this repo (or at least `training/`, `data/`, `kaggle/`) to the
     notebook filesystem (unzip a zip of the repo into `/kaggle/working`).

## Run training
    !python kaggle/train_image.py

Defaults used by the canonical script:
- image dir: auto-detected Kaggle mount (`/kaggle/input/crop-yield-...`)
- output: `/kaggle/working/outputs/image`
- archs: `efficientnet_v2_s,convnext_tiny` (benchmarked; chosen on val only)
- two-stage: head (frozen backbone, lr 3e-3, 8 epochs) → finetune (lr 1e-4, 8)
- early stop on **validation** macro F1, patience 3; test never used
- weights: `cap10` scheme from Phase 3; augmentation only on train
- pretrained = True (ImageNet weights); input channels = discovered bands;
  `C != 3` stems adapted via the documented Kratzert mean-init

Expected outputs written to `/kaggle/working/outputs/image`:
`best_image_model.pt`, `efficientnet_v2_s.pt`, `convnext_tiny.pt`,
`best_standalone_image.json`, `image_config.json`, `preprocessing.json`,
`coverage_report.json`, `leakage_check.json`, `image_index.csv`.

## Evaluate ONCE on the test split (after training, still the same seed/config)
    !python kaggle/train_image.py --evaluate --final-test

Writes `results_test.json`. The test set (Sullia) must receive exactly one
evaluation after selection.

## Copy results back
Download `/kaggle/working/outputs/image` → local `models/image/`.

## Record the reproducibility block
In `docs/PHASE_4_IMAGE_RESULTS.md` fill in: notebook URL + run id, date,
accelerator, Kaggle dataset version (10), pytorch/timm versions printed at the
top of the run, exact command line, checksum of the model zip, and the
dataset inventory tree/truncated listing the wrapper prints.

## Command-line reference (canonical scripts, identical locally and on Kaggle)
    python training/train_image.py \
        --repo-root . --image-dir <dir> --output models/image \
        [--archs efficientnet_v2_s,convnext_tiny] [--img-size 128] [--patch-size 64]
        [--batch-size 64] [--epochs-head 8] [--epochs-finetune 8] [--early-stop 3]
        [--seed 42] [--weight-scheme cap10] [--stats-samples 2500]
    python training/evaluate_image.py --repo-root . --output models/image --final-test

Local smoke (already passing, synthetic rasters, CPU): `--no-pretrained`,
`--max-train-samples 32`, `--epochs-head 2 --epochs-finetune 1 --workers 0`.