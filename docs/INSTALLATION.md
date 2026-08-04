# Installation

Local (non-Docker) setup for development and research. For a production-like
stack, prefer [DEPLOYMENT.md](DEPLOYMENT.md).

## Prerequisites

- Python 3.12+
- Node.js 20+ and npm
- (Recommended) conda or venv for isolation

## 1. Create the environment

### conda

```bash
conda env create -f environment.yml
conda activate cropfusion
```

### venv

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

## 2. Install first-party packages

```bash
pip install -e ./ai/models ./ai/preprocessing ./ai/training ./ai/explainability
pip install -e ./services/dataset_manager ./services/spatial_alignment
```

## 3. Frontend

```bash
cd frontend
npm ci
cd ..
```

## 4. Verify

```bash
make test          # full Python test suite
make compose       # validates docker-compose files (requires Docker)
cd frontend && npm test && cd ..
```

## Optional services

For a full local backend (Postgres/PostGIS, Redis), run the data services
only:

```bash
docker compose up -d postgres redis
```

The backend defaults to SQLite for tests; point `BACKEND_DATABASE__URL` at the
Postgres instance for integration work.

## Windows notes

- GDAL/rasterio wheels are prebuilt; use the conda environment if pip wheels
  fail to install.
- `torch.compile` requires a C compiler; CropFusion automatically falls back
  to eager execution when one is unavailable.
