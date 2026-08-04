# Changelog

All notable changes to CropFusion are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026

### Added
- **Deployment**: Docker Compose dev + prod stacks (frontend, backend, Postgres/
  PostGIS, Redis, inference replica, admin scheduler, docs), Caddy TLS edge,
  Prometheus/Grafana/Loki/Promtail/node-exporter observability stack.
- **CI/CD**: GitHub Actions workflows for lint, tests, coverage, Docker builds
  (GHCR), security scans (Safety/Trivy/Bandit/npm audit/CodeQL), releases and
  deployments. Dependabot enabled.
- **MLOps** (`mlops/`): filesystem model registry with `draft -> staging ->
  production` lifecycle, promotion gates (metrics, regression, drift,
  fairness), rollback, experiment tracking, scheduled monitoring, release
  reports, `cropfusion-mlops` CLI.
- **Reproducibility**: pinned `requirements.txt`, `environment.yml`,
  umbrella `pyproject.toml`, `VERSION`.
- **Backups**: DB + assets backup/restore scripts with S3 offsite support.
- **Security**: env templates, secrets handling, security policy, hardening.
- **Research**: architecture diagrams, model architecture, benchmark and
  training/eval references, dataset statistics generator.
- **Documentation**: full guide set, user manuals, website content, release
  package (Apache-2.0 LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, templates).

### Changed
- Project standardized at version **1.0.0**.

### Fixed
- (From Phase 11) Prometheus middleware double in-flight decrement; optimized
  runtime compiled-mode fallback to eager execution.

## [0.1.0] - 2026

### Added
- Phases 1-11: dataset management, spatial alignment, preprocessing,
  multimodal models, training, explainability, enterprise FastAPI backend,
  frontend SPA, ML quality (drift/fairness/monitoring/optimization),
  observability.
