# Configuration guide

Everything is configurable through YAML (env > YAML > defaults). Generate a
template with:

```python
from ai.models import save_model_template
save_model_template("configs/model.yaml")
```

Or load/save at runtime:

```python
from ai.models import load_model_config, ModelConfig
cfg = load_model_config("configs/model.yaml")
cfg = ModelConfig.from_preprocessor(preprocessor, temporal={"max_len": 16})
```

## Section reference

### `name` / `version`

Model identity (stored in checkpoints for provenance).

### `tabular` — TabTransformer

| Key | Default | Options / notes |
|-----|---------|-----------------|
| `numeric_dim` | 0 | Leading continuous features. |
| `categorical_cardinalities` | `[]` | Cardinality per categorical slot (ordinal codes). |
| `embedding_dim` | 64 | Token width. |
| `depth` | 4 | Transformer blocks. |
| `num_heads` | 4 | Must divide `embedding_dim`. |
| `ff_dim` | 256 | Feed-forward width. |
| `dropout` | 0.1 | Dropout probability. |
| `activation` | `gelu` | `relu` `gelu` `silu` `tanh` `leaky_relu`. |
| `use_cls` | true | Prepend/pool a CLS token. |
| `position_encoding` | `none` | `none` `sinusoidal` `learned`. |
| `max_len` | 64 | Upper bound on feature tokens. |

### `image_encoder` — timm backbones (shared)

| Key | Default | Notes |
|-----|---------|-------|
| `backbone` | `efficientnetv2_s` | Any timm name: ConvNeXt, ResNet50, Swin, ... |
| `pretrained` | false | ImageNet weights (requires network first run). |
| `freeze_backbone` | false | Freeze both encoders after build. |
| `input_size` | `null` | Square patch edge; `null` = backbone native resolution. |
| `channel_expansion` | `repeat` | `repeat` or `conv` (1→3 channels). |
| `drop_path_rate` | 0.0 | Stochastic depth. |
| `ndvi_backbone` / `evi_backbone` | `null` | Per-modality overrides. |

### `image_fusion` — NDVI/EVI fusion

| Key | Default | Notes |
|-----|---------|-------|
| `method` | `learnable` | `concat` `weighted_sum` `learnable` `attention`. |
| `hidden_dim` | `null` | Output width; `null` = encoder width. |
| `dropout` | 0.1 | |

### `temporal` — TemporalTransformer

| Key | Default | Notes |
|-----|---------|-------|
| `d_model` | 256 | Transformer width. |
| `depth` | 2 | Blocks. |
| `num_heads` | 4 | Must divide `d_model`. |
| `ff_dim` | 1024 | Feed-forward width. |
| `dropout` | 0.1 | |
| `activation` | `gelu` | |
| `use_cls` | true | CLS pooling. |
| `position_encoding` | `learned` | `none` `sinusoidal` `learned`. |
| `max_len` | 16 | **Must cover** `temporal.max_observations`. |
| `embedding_dim` | 256 | Output image-embedding width. |

### `cross_attention` — Q=image, K=V=tabular

`num_heads` (4), `dropout` (0.1), `out_dim` (256).

### `gated_fusion` — AdaptiveGatedFusion

`out_dim` (256), `hidden_dim` (256), `dropout` (0.1).

### `shared_encoder` — shared latent space

`d_model` (256), `depth` (2), `num_heads` (4), `ff_dim` (1024), `dropout`
(0.1), `activation` (`gelu`), `out_dim` (512 — try 768 / 1024).

### `heads` — task heads

| Key | Default | Notes |
|-----|---------|-------|
| `crop.num_classes` | 0 | Set from the Phase 4 label encoder; 0 disables the head. |
| `crop.hidden_dim` / `activation` / `dropout` | null / `relu` / 0.1 | |
| `yield_prediction.hidden_dim` / `activation` / `dropout` | null / `relu` / 0.1 | |
| `yield_prediction.output_clamp_min` | null | Lower clamp (e.g. 0.0) on predicted yield. |

### `loss` — loss interfaces (Phase 6 uses these)

| Key | Default | Notes |
|-----|---------|-------|
| `crop_loss` | `label_smoothing` | `cross_entropy` `label_smoothing` `focal`. |
| `yield_loss` | `huber` | `mse` `huber`. |
| `crop_weight` / `yield_weight` | 0.7 / 0.3 | Fixed weighting. |
| `weighting_mode` | `fixed` | `fixed` or `learnable` (Kendall). |
| `label_smoothing` | 0.1 | |
| `focal_gamma` | 2.0 | |
| `reduction` | `mean` | |

### `checkpoint` — CheckpointManager defaults

`directory` (`artifacts/models`), `keep_last` (3).

### `export` — export defaults

`onnx_opset` (17), `torchscript_mode` (`trace`).

### `validate_inputs`

`true` — run shape/dtype validation on every forward pass. Disable for
maximum inference throughput once shapes are known-good.

## Deriving the config from Phase 4

`ModelConfig.from_preprocessor(preprocessor, **overrides)` fills in:

* `tabular.numeric_dim` and `categorical_cardinalities` from the fitted
  tabular pipeline (ordinal encoding) — or all-continuous when one-hot.
* `heads.crop.num_classes` from the label encoder.
* `image_encoder.input_size` from the preprocessor image size.
* `temporal.max_len` from `max_observations`.

Example:

```python
model = ModelFactory.from_preprocessor(
    preprocessor,
    image_encoder={"backbone": "convnext_tiny", "pretrained": False},
    shared_encoder={"out_dim": 768},
)
```

## Environment resolution

`MODEL_CONFIG_FILE` points at a YAML; individual values can be overridden with
`MODEL_<SECTION>_<KEY>` env vars (e.g. `MODEL_SHARED_ENCODER_OUT_DIM=1024`).
