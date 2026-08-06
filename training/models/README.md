# CropFusion AI Models (Phase 5)

The complete multimodal neural architecture for **crop recommendation** and
**yield prediction**, consuming the AI-ready tensors produced by
[Phase 4](../preprocessing/).

> **Scope.** This phase implements the *architecture only* — forward pass,
> config, factory, checkpointing and export. There is **no training code**,
> optimizer, scheduler or inference API; those belong to Phase 6.

---

## Input contract (Phase 4 batch)

The model consumes exactly the batch dict produced by
`training.preprocessing.dataloader.collate_samples`:

| Key | Shape | Dtype | Meaning |
|-----|-------|-------|---------|
| `tabular` | `[B, F]` | float32 | continuous features then categorical ordinal codes |
| `ndvi` | `[B, T, 1, H, W]` | float32 | NDVI patch sequence |
| `evi` | `[B, T, 1, H, W]` | float32 | EVI patch sequence |
| `temporal_mask` | `[B, T]` | float32 | 1 = real observation, 0 = padding |

`T` is `temporal.max_observations`, `H = W` is the patch size. For the
TabTransformer to receive categorical indices the Phase 4 preprocessor should
use `categorical_encoding: ordinal` (see `ModelConfig.from_preprocessor`).

## Architecture

```
tabular [B, F]                    ndvi [B,T,1,H,W]        evi [B,T,1,H,W]
      │                                   │                      │
 TabTransformer                       NdviEncoder           EviEncoder
      │                             (timm backbone)      (timm backbone)
 tabular embedding                        │                      │
      │                                 └──────┬───────────────┘
      │                                        ▼
      │                            ImageFusion (concat / weighted / learnable / attention)
      │                                        │
      │                                        ▼
      │                             TemporalTransformer (+ CLS + positional enc)
      │                                        │
      │                               image embedding
      │                                        │
      └──────────────┬─────────────────────────┘
                     ▼
        CrossAttention (Q=image, K=V=tabular)
                     ▼
         AdaptiveGatedFusion (image / tabular / fusion gates)
                     ▼
         SharedMultimodalEncoder (shared latent space)
                     ▼
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
   CropHead (softmax)                              YieldHead (regression)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/LAYER.md](docs/LAYER.md) and [docs/TENSOR_FLOW.md](docs/TENSOR_FLOW.md).

## Quickstart

```python
from training.models import ModelFactory

# Build directly from a fitted Phase 4 preprocessor (derives the tabular
# schema, class count and image size automatically).
model = ModelFactory.from_preprocessor(preprocessor)

batch = model.sample_batch(batch_size=4)
output = model(batch)          # CropFusionOutput
output.crop_logits             # [B, num_classes] logits
output.yield_pred              # [B, 1] predicted yield
output.gates                   # {image_gate, tabular_gate, fusion_gate} in [0,1]
```

Build from YAML:

```python
from training.models import ModelFactory
model = ModelFactory.from_config_file("configs/model.yaml")
model.save_config("configs/model.yaml")      # or save a template first
```

## Factory, checkpointing, export

```python
from training.models import ModelFactory, CheckpointManager, ModelExporter

# Factory
ModelFactory.create(config)                    # from ModelConfig / dict
ModelFactory.from_config_file(path)            # from YAML
ModelFactory.from_preprocessor(preprocessor)   # from Phase 4
ModelFactory.freeze_layers(model, [r"backbone\."])
ModelFactory.load_pretrained(model, ckpt_path, strict=False)

# Checkpointing
ckpt = CheckpointManager("artifacts/models", keep_last=3)
ckpt.save(model, epoch=10, metrics={"loss": 0.4})
resumed = ckpt.resume(path, model=model, optimizer=optimizer, scheduler=scheduler)
report = CheckpointManager.partial_load(model, path, include=[r"ndvi_encoder\."])

# Export
exporter = ModelExporter(model)
exporter.export_torchscript("model.ts")
exporter.export_onnx("model.onnx")            # needs `pip install onnx`
```

## Tests

```bash
python -m pytest training/models/tests
```

## Documentation

* [Architecture diagram](docs/ARCHITECTURE.md)
* [Layer diagram & per-layer specs](docs/LAYER.md)
* [Tensor flow](docs/TENSOR_FLOW.md)
* [Module documentation](docs/MODULES.md)
* [Configuration guide](docs/CONFIGURATION.md)
* [Developer guide](docs/DEVELOPER.md)
