# Developer guide

## Repo conventions

* Python 3.12+, PyTorch 2.x, timm ≥ 1.0.
* `from __future__ import annotations`; full type hints; docstrings with
  `Args:` / `Returns:` / `Raises:`.
* Errors raise `ModelError` subclasses with `MDL-<AREA>-<NNN>` codes
  (see `exceptions.py`).
* No TODOs, no placeholders, no dead code. If a module is present it is used.

## Running tests

```bash
python -m pytest ai/models/tests -q
```

The suite exercises forward passes, shapes, gradients, checkpoints, config and
export. ONNX tests skip when the `onnx` package is absent; the TensorRT
entry point is asserted to raise `MissingDependencyError`.

## Adding a task head

```python
model.add_head("crop_health", MyHealthHead(in_dim=shared_dim, ...))
```

Heads register through `MultiTaskHeads` and receive the shared representation;
training (Phase 6) reads them via `model.heads.names` and
`model(heads)` output keys. No architectural change needed.

## Switching fusion methods

`image_fusion.method: concat | weighted_sum | learnable | attention`. Each
method is a self-contained branch in `image_fusion.py`; the output width
(`out_dim`) stays constant so the rest of the pipeline is unaffected.

## Swapping backbones

`backbone` accepts any timm model that pools to a vector
(e.g. `efficientnetv2_s`, `convnext_tiny`, `resnet50`, `swin_tiny_patch4_window7_224`).
`TimmImageEncoder` probes the real output width at construction because
`timm.num_features` under-reports for some backbones.

Note: BatchNorm-based backbones require batch ≥ 2 in train mode; the 
architecture probe and exporter run in eval mode for this reason.

## Freezing / transfer learning

```python
ModelFactory.freeze_layers(model, [r"^tabular\.", r"^shared_encoder\."])
ModelFactory.freeze_backbone(model)               # freeze NDVI + EVI timm nets
ModelFactory.load_backbone(model, ckpt_path)      # backbone-only partial load
ModelFactory.load_pretrained(model, ckpt_path)    # full partial load
```

Freezing happens after construction so the config's `freeze_backbone` and
`load_*` calls compose.

## Export notes

* TorchScript uses **tracing** (the dict-based `forward` is not scriptable).
  `forward_export` is the traced entry point.
* ONNX requires `pip install onnx`; dynamic batch and time axes are enabled.
* TensorRT is a planned Phase 6+ deployment target — `export_tensorrt` raises
  `MissingDependencyError` until the runtime is available (documented, not a
  no-op).

## Design constraints (do not violate)

* The model consumes **only** the Phase 4 batch contract
  (`tabular`, `ndvi`, `evi`, `temporal_mask`).
* No training / optimizer / scheduler / inference-API code in this package.
* Phase 4 tensor layout is `[continuous | categorical ordinal codes]`; the
  tabular feature width must equal
  `numeric_dim + len(categorical_cardinalities)`.

## Versioning

`__version__` in `ai/models/__init__.py`; config carries `version` for
checkpoint provenance. Bump both when the architecture changes.
