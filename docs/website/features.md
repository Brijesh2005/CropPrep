# Features

Marketing-focused summary of CropFusion capabilities for the website.

## Prediction

- Crop recommendation with ranked alternatives and confidence scores.
- Yield prediction with expected value and per-crop breakdown.
- Year + season selection (Kharif / Rabi / ...).

## Explainability

- Plain-language explanations of every recommendation.
- Top contributing features (rainfall, temperature, vegetation index, ...).
- Per-prediction attribution records for transparency and audit.

## Spatial intelligence

- Map-based field selection (drop a pin or pick a saved location).
- District / village boundary resolution and nearest-location lookups.
- GIS backed by PostGIS.

## Dataset & model management

- Dataset ingestion with profiling, validation, checksums and versioning.
- Model registry with `draft -> staging -> production` lifecycle.
- Promotion gates: accuracy, latency regression, drift, fairness.
- One-command rollback.

## ML quality monitoring

- Drift detection across features, labels, predictions, spatial and temporal
  dimensions.
- Fairness evaluation across protected groups and regions.
- Prometheus exporters + Grafana dashboards (ML quality, performance).

## Operations

- Docker Compose dev + prod deployments with automatic TLS (Caddy).
- Structured JSON logs, OpenTelemetry tracing, Loki aggregation.
- Automated backups (DB + assets) with offsite S3 support.
- CI/CD with linting, tests, coverage, security scans and releases.
