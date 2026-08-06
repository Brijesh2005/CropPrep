# Model summary, architecture report & metadata

`CropFusionModel.summary()` gives a complete picture of a built model:
parameters, per-layer detail, real traced input/output shapes and memory
estimates — everything needed to review an architecture or size a deployment.

## `model.summary(sample_batch=None)`

| key | content |
| --- | --- |
| `config` | the full serialised `ModelConfig` |
| `metadata` | `model.metadata` (architecture info, versions, dims) |
| `output_names` | enabled task-head names (`crop` / `yield` / …) |
| `parameter_summary` | `{total, trainable, frozen}` |
| `parameter_count` / `trainable_parameters` | quick numbers |
| `layer_summary` | one row per named submodule (type, params, trainable) |
| `architecture_report` | **per-module real shapes** (only with `sample_batch`) |
| `input_shapes` / `output_shapes` | batch shapes in → head outputs out |
| `memory_estimate` | parameter / activation bytes and MB |

Without `sample_batch` the shape-dependent keys are `None`. With it, the model
is run once (eval, `no_grad`) to trace shapes, so the report reflects the real
configured widths — e.g. for the shared test config:

```json
{
  "input_shapes":  {"tabular": [4, 5], "ndvi": [4, 4, 1, 32, 32],
                    "evi": [4, 4, 1, 32, 32], "temporal_mask": [4, 4]},
  "output_shapes": {"crop_logits": [4, 3], "yield_pred": [4, 1],
                    "shared_representation": [4, 128]}
}
```

## `architecture_report`

`training/models/utils.py::architecture_report(module, forward_fn)` is the
standalone helper behind the summary. It registers forward hooks on every named
submodule, runs one forward, and joins the captured input/output shapes with
parameter counts. It understands nested tuples, dicts and dataclasses (e.g.
`FusionOutput`), so multi-output components are fully described.

```python
from training.models.utils import architecture_report

rows = architecture_report(model, forward_fn=lambda: model(batch))
# rows[0] == {"name": ..., "type": ..., "params": ..., "trainable": ...,
#             "input_shapes": [...], "output_shapes": ...}
```

This is the tool for printing a Keras-style shape table of the whole model,
including the `CrossModalFusionEngine` internals (`fusion_engine.shared_encoder`
etc.).

## `model.metadata`

A small JSON-safe dict stored in checkpoints and logs:

```python
model.metadata
# {"name": "cropfusion_v1", "version": "1.0.0",
#  "architecture_version": "1.0.0", "output_names": ["crop", "yield"],
#  "embedding_dims": {"tabular": ..., "image": ...}, "shared_dim": 128,
#  "precision": "float32", "gradient_checkpointing": false,
#  "uses_cross_attention": true, "uses_gated_fusion": true,
#  "pytorch_version": "2.13.0+cpu", "python_version": "3.12"}
```

See `ARCHITECTURES.md` for how metadata participates in checkpoint rebuilds.
