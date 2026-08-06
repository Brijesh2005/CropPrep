# Architecture registry & version management

`ModelFactory` centralises construction and keeps a **registry of model
architectures** so a config — or a checkpoint — can be rebuilt with the right
model class, and so future architectures plug in without touching the built-in
code.

## The registry

```python
ModelFactory._ARCHITECTURES  # {"cropfusion_v1": CropFusionModel, ...}
```

- `cropfusion_v1` → `CropFusionModel` is the built-in architecture.
- `ModelFactory.register_architecture(name, model_cls)` adds a future
  architecture. `model_cls` must be an `nn.Module` subclass whose constructor
  accepts a `ModelConfig`; anything else raises `MDL-CONFIG-001`.
- `ModelFactory.architecture_names()` lists registered names.

## Resolution rules

| call | behaviour |
| --- | --- |
| `create(config)` | resolves via `config.name`; falls back to `CropFusionModel` for unregistered display names |
| `create(config, architecture="cropfusion_v2")` | requires a **registered** name; unknown names raise `MDL-CONFIG-001` |
| `from_checkpoint(path)` | rebuilds with the checkpoint's stored `architecture` (falling back to its config name) |

The fallback for unregistered config names keeps old YAML files working, while
the explicit `architecture=` argument gives strict, version-safe construction.

## Version management

`ModelConfig.architecture_version` (default `"1.0.0"`) is a **schema version of
the architecture** the config describes. It is bumped on breaking architectural
changes. It is stored in checkpoints and in `model.metadata`, so a deployer can
validate that a checkpoint belongs to the same architecture before loading.

## Checkpoint integration

`CheckpointManager.save` stores alongside the weights:

```yaml
architecture:          cropfusion_v1
architecture_version:  1.0.0
metadata:              {name, version, architecture_version, output_names,
                        embedding_dims, shared_dim, precision, ...,
                        pytorch_version, python_version}
model_config:          {full config, JSON-safe}
```

`metadata` is JSON-serialised on save so `torch.load(weights_only=True)` can
read it (`torch.__version__` is a `TorchVersion` object and would otherwise be
rejected by the safe loader).

```python
from training.models import CheckpointManager, ModelFactory

path = CheckpointManager("artifacts/models").save(model, epoch=3)
restored = ModelFactory.from_checkpoint(path)   # same class, weights restored
assert restored.metadata["architecture_version"] == model.metadata["architecture_version"]
```

## `model.metadata`

Every `CropFusionModel` exposes a `metadata` dict: config name/version,
`architecture_version`, enabled `output_names`, per-modality embedding widths,
`shared_dim`, current precision / checkpointing flags, and the PyTorch /
Python versions used at build time. `model.summary()` embeds it under the
`metadata` key.
