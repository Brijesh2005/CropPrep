# Shared Framework — Architecture

How `shared/` fits into the CropFusion monorepo and why the dependency rules
matter.

## Platform layout

```text
D:\CropPrep\
├── shared\            reusable, platform-agnostic contract layer (R1.3)
├── training\          Training Platform (dataset manager, STAM, preprocessing,
│                      models, training loop, explainability, mlops, kaggle)
├── application\       Prediction Platform (FastAPI backend + React frontend)
├── docs\              guides, diagrams, migration reports
└── (deployment, data, notebooks, scripts, ...)
```

## Dependency rules

The monorepo is split into three layers with strictly one-directional
dependencies:

```text
┌──────────────┐     ┌───────────────────┐
│  training/   │     │   application/    │
│  (Training   │     │   (Prediction     │
│   Platform)  │     │    Platform)      │
└──────┬───────┘     └─────────┬─────────┘
       │                       │
       │   depends only on     │
       ▼                       ▼
┌──────────────────────────────────────┐
│              shared/                  │
│  config · enums · exceptions ·        │
│  interfaces · logging · schemas ·     │
│  serialization · types · utils ·      │
│  validation · versioning              │
└──────────────────────────────────────┘
          │ (stdlib + third-party only)
          ▼
   Python stdlib / site-packages
```

- `training/` never imports `application/`.
- `application/` never imports `training/` **for shared utilities**.
- `shared/` imports only the standard library and third-party packages
  (pydantic, PyYAML, numpy/torch/pandas lazily inside serializers).

## Why

Before R1.3 the same helpers were copy-pasted across
`training/dataset_manager/config.py`, `training/training/config.py`,
`training/stam/config.py`, `training/preprocessing/config.py`,
`training/explainability/config.py` and `application/backend/app/core/config.py`.
Fixes had to be applied in up to six places, and the backend reached *into*
`training` for private helpers (`_yaml_safe`), coupling the two platforms.

R1.3 extracted those helpers into `shared/` so:

1. **One source of truth** — a config/env/yaml fix is applied once.
2. **No private cross-platform imports** — the backend imports `shared.*`,
   never `training.*` internals.
3. **Portable vocabulary** — both platforms share `CropType`, `Season`,
   `Severity`, `DatasetStatus`, etc. through `shared.enums` instead of
   redefining them.
4. **Interchange contracts** — `shared.schemas` and `shared.interfaces`
   describe how platforms exchange datasets, images, predictions and
   validation reports without depending on each other's internals.

## Known coupling (deliberate, out of R1.3 scope)

`application/` still *delegates execution* to Training Platform algorithm
modules (e.g. `training.models.ModelFactory`, `training.stam.STAM`) for real
prediction work. That is intentional composition, not utility reuse, and
untangling it would require relocating the training algorithms — which R1.3
explicitly leaves untouched. See the [migration report](../migration/MIGRATION_REPORT_R1.3.md)
for details.

## Diagrams

- [R1.3 package layout](../diagrams/r1-3-shared-packages.md)
- [R1.3 dependency rules](../diagrams/r1-3-dependency-rules.md)
- [R1.3 configuration resolution](../diagrams/r1-3-config-resolution.md)
- [R1.3 serialization & validation](../diagrams/r1-3-serialization-validation.md)
- [R1.3 migration flow](../diagrams/r1-3-migration-flow.md)
