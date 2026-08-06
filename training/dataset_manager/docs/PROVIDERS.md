# Provider Registry & Multi-Provider Guide

R2.2 introduces a **provider registry**: the single place every data source is
registered and resolved. The Dataset Manager no longer talks to providers
directly — consumers go through `DatasetManager.provider_registry`, which owns
registration, priority ordering, health, availability and capabilities.

## Architecture

```
DatasetManager ──► ProviderRegistry ──► GitRepositoryTabularProvider
                  (registry)      └──► KaggleHubImageProvider
                  │                    └──► future plugins
                  ├─► resolve(name)            — enabled instance by name
                  ├─► resolve_by_kind(kind)    — enabled providers, priority order
                  ├─► availability()           — {name: bool}
                  ├─► health()                 — per-provider health snapshot
                  ├─► capabilities()           — per-provider capability manifest
                  └─► discovery()              — plain registration records
```

Two providers are registered by default:

| Name | Kind | Role |
| --- | --- | --- |
| `git_repository_tabular` | `tabular` | Git-versioned CSVs (discover / load / schema / statistics / join) |
| `kaggle_hub_image` | `image` | Kaggle Sentinel NDVI/EVI (download-or-reuse, catalog, lazy reads, patches) |

## Registry semantics

* **Resolution honours `enabled`** — resolving a disabled provider raises
  `DatasetNotFoundError` (`DM-FIND-001`).
* **Priority orders resolution** — `resolve_by_kind` returns providers sorted by
  priority (highest first); equal priorities fall back to registration order.
* **The manager never holds providers directly** — the legacy
  `tabular_provider` / `image_provider` attributes are wired through the
  registry (bypassing the enabled check so a disabled provider is still
  constructible) to keep the R1.2 surface working.

## Configuration

Provider behaviour is configured under `providers:` in the settings (YAML, env
or pydantic `Settings`):

```yaml
providers:
  tabular:
    root: training/datasets/tabular
  image:
    catalog_name: kaggle-crop-yield
  registry:
    providers:
      # Override a default provider.
      - name: kaggle_hub_image
        kind: image
        enabled: false            # disabled, not removed
      - name: kaggle_hub_image
        kind: image
        enabled: true
        priority: 100
      # Register an additional provider (future plugins).
      - name: aux_tabular
        kind: tabular
        enabled: true
        priority: 50
        config:
          root: /data/aux
          patterns: ["*.csv"]
```

* `enabled: false` disables without removing — the provider stays visible to
  `availability()` / `health()` / `discovery()` but cannot be resolved.
* `priority` controls shadowing among providers of the same kind.
* `config` is forwarded to the provider factory for additional entries.

## Python API

```python
manager = DatasetManager(settings)

# Introspection
manager.provider_manifests()      # {name: manifest} for diagnostics
manager.availability()            # {"kaggle_hub_image": True, ...}
manager.health()                  # per-provider health snapshots
manager.discovery()               # plain registration records

# Registry access
registry = manager.provider_registry
registry.resolve("git_repository_tabular")     # enabled instance (or raises)
registry.resolve_by_kind("tabular")            # [GitRepositoryTabularProvider]
registry.names()                               # registered names
registry.registrations()                       # enabled + disabled
```

## CLI

```bash
python -m training.dataset_manager providers        # manifests
python -m training.dataset_manager health           # health snapshots
python -m training.dataset_manager availability     # {name: available}
python -m training.dataset_manager discovery        # registration records
```

## Persistence

Provider registrations, status, capabilities and availability are persisted to
the `provider_metadata` table of `metadata.db` — see
[METADATA_REPOSITORY.md](METADATA_REPOSITORY.md).
