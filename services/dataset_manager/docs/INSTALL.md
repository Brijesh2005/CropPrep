# Installation Guide

## Prerequisites

* **Python 3.12+** (the project targets 3.12+).
* **Pip** (or uv / poetry).
* No GDAL system install required — `rasterio` ships binary wheels.

## 1. Install dependencies

```bash
pip install -r services/dataset_manager/requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `kagglehub` | automatic Kaggle dataset download |
| `pandas`, `numpy` | CSV profiling / raster arrays |
| `rasterio` | GeoTIFF header + windowed reads |
| `pydantic` | validated configuration |
| `PyYAML` | YAML configuration files |
| `pyarrow` (optional) | metadata export to Parquet |
| `pytest` | test runner |

> `rasterio` needs a compatible `numpy`. Install them together if your
> environment already pins numpy.

## 2. Verify the installation

```bash
python -c "from services.dataset_manager import DatasetManager; print('ok')"
```

## 3. Optional: generate a config template

```bash
python services/dataset_manager/manage_dataset.py config-template dm.yaml
```

## 4. Kaggle authentication (first download only)

`kagglehub` downloads **public** datasets without credentials. If the primary
dataset ever becomes private, set the `KAGGLE_USERNAME` / `KAGGLE_KEY`
environment variables or run `kagglehub.login()` once.

## 5. Run the test-suite (development)

```bash
cd services/dataset_manager
pytest
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DM_DATASET_ROOT` | `./datasets` | Managed dataset root |
| `DM_CONFIG_FILE` | — | YAML configuration path |
| `DM_DOWNLOAD__KAGGLE_HANDLE` | `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada` | Primary dataset |
| `DM_DOWNLOAD__FORCE_DOWNLOAD` | `false` | Always re-download |
| `DM_SCAN__WORKERS` | `8` | Scanner thread count |
| `DM_CACHE__ENABLED` | `true` | Enable scan/inventory cache |
| `DM_LOG__LEVEL` | `INFO` | Logging level |
