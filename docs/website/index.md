# CropFusion - Landing Page

> Content for the public website. Adapt to the site framework as needed.

## Hero

**Grow with confidence.**

CropFusion turns satellite imagery and agronomic data into clear, explainable
crop recommendations and yield predictions - so every field decision is backed
by evidence.

- Multimodal AI (tabular + NDVI/EVI satellite indices)
- Explainable recommendations in plain language
- Works at district and village resolution
- Open source, self-hosted, MLOps-ready

## Problem

Smallholder and district-level farmers face huge uncertainty: weather shifts,
soil variability, and limited access to agronomic expertise. Most decision
support is either too generic or requires experts to interpret.

## Solution

CropFusion fuses **tabular agronomic features** (soil, rainfall, temperature,
irrigation) with **satellite vegetation indices** through a transformer-based
multimodal model. It predicts:

- **Which crop to plant** - a ranked recommendation with confidence.
- **What yield to expect** - per crop, per location, per season.
- **Why** - an explanation of the driving factors, understandable without
  technical background.

## Features

See [features.md](features.md) for the full list.

## Architecture

Deploy anywhere with Docker Compose:

- FastAPI backend + PostGIS + Redis
- React PWA frontend (works offline)
- Prometheus + Grafana + Loki observability
- Model registry with promotion gates, drift and fairness monitoring

## Open source

CropFusion is Apache-2.0 licensed. Contribute, self-host, or build on it.

- [GitHub](https://github.com/{owner}/cropfusion)
- [Documentation](../../README.md)
- [Security policy](../../SECURITY.md)
