# CropFusion Inference-Only Deployment Guide

R1.4 adds a **standalone inference image** that ships the Prediction Platform
without the Training Platform. It loads exported artifacts only.

## Why a standalone image

`application/docker/Dockerfile.backend` bundles `training/*` because the API
still embeds the training-side pipeline. The inference-only process should not
ship training code at all — it consumes `application/inference_package` +
`application/models`, which is smaller, faster to build and less attack
surface.

## The image

`application/docker/Dockerfile.inference.standalone`

- Base: `python:3.12-slim`
- Copies: `shared/`, `application/inference/`, `application/inference_package/`,
  `application/models/`, `application/gis/`, `application/history/`,
  `application/config/`, `application/database/`
- Excludes: `training/` (no `pip install` of training packages)
- `PYTHONPATH=/app/application`

Build:

```bash
docker build -f application/docker/Dockerfile.inference.standalone \
    -t cropfusion/inference-standalone .
```

R1.4 entrypoint is a placeholder that imports every inference-layer package and
prints its version — proof the image has no `training` dependency. The real
serving entrypoint ships in a later phase.

## Running with exported artifacts

Model weights and the inference package are mounted read-only:

```bash
docker run --rm \
  -v "$(pwd)/application/models:/app/application/models:ro" \
  -v "$(pwd)/application/inference_package:/app/application/inference_package:ro" \
  cropfusion/inference-standalone
```

## Config

The future process reads the multi-YAML templates in `application/config/`
(`application.yaml`, `model.yaml`, `inference.yaml`, `logging.yaml`,
`security.yaml`, `database.yaml`) through `shared.config.load_yaml_config` with
precedence:

```
env (CF_<SECTION>__<KEY>) > YAML > defaults
```

Production secrets are never in YAML; they arrive via `CF_SECURITY__SECRET_KEY`
and friends.

## Comparison

| | `Dockerfile.backend` | `Dockerfile.inference.standalone` |
| --- | --- | --- |
| Ships `training/` | yes | no |
| Ships exported artifacts | checkpoint via config | mounted package + weights |
| Roles | API + embedded inference | dedicated inference worker (future) |
| R1.4 entrypoint | uvicorn app | placeholder verification |

See also the deployment diagram: [r1-4-deployment-flow](../diagrams/r1-4-deployment-flow.md).
