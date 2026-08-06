# Model Exporter (Phase R5)

`training.inference.ModelExporter` wraps the Phase-5 exporter (TorchScript
trace + ONNX) and adds the **PyTorch format** the inference package is built
around: a self-describing `.pt` file holding the config, the state dict and
export metadata, so a consumer can rebuild the architecture and load the
weights without the training package.

## Formats

| format | file | notes |
| --- | --- | --- |
| `pytorch` | `cropfusion.pt` | `{"format", "format_version", "config", "state_dict", "metadata"}` payload |
| `torchscript` | `cropfusion.torchscript.pt` | traced module (Phase 5 exporter) |
| `onnx` | `cropfusion.onnx` | opset `ExportConfig.onnx_opset` (default 17), dynamic batch |

## Usage

```python
from training.inference import InferenceConfig, ModelExporter, load_pytorch_model

config = InferenceConfig(exporter={"formats": ["pytorch", "onnx"]})
paths = ModelExporter(model, sample_batch=None).export_bundle("out", config=config)

# sidecars written by export_bundle:
#   model_config.yaml · metrics.json · metadata.json · checksums.json

model, metadata = load_pytorch_model("out/cropfusion.pt")
```

`export_bundle` returns a mapping of artifact key → path. `metadata.json`
records the package/model/dataset semver, the model fingerprint, the git
commit and the produced formats. `checksums.json` holds the SHA-256 of every
written file.

`load_pytorch_model` validates the payload is a `cropfusion-pytorch` export,
rebuilds the model from the embedded config via `ModelFactory` and returns the
model in `eval()` mode together with the metadata.

## Exporter configuration

```yaml
exporter:
  formats: [pytorch, torchscript, onnx]
  onnx_opset: 17
  torchscript_mode: trace
  export_batch_size: 2
```

Env override: `INF_EXPORTER__FORMATS='["pytorch"]'` (lists are JSON), etc.
