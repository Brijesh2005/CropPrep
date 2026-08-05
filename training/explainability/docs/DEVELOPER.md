# Developer Guide

## Architecture

```
ai/explainability (ports: interfaces.py)
   ├── config.py            ExplainabilityConfig — one section per explainer
   ├── utils.py             AttentionCapture, GradCAM target discovery, helpers
   ├── shap_explainer.py    self-contained KernelSHAP + gradient SHAP
   ├── gradcam.py           ImageExplainer + CAM methods
   ├── integrated_gradients.py
   ├── temporal_attention.py
   ├── cross_modal_attention.py
   ├── uncertainty.py       confidence / MC-dropout / ECE
   ├── counterfactual.py
   ├── visualization.py     matplotlib figures
   ├── exporter.py          HTML / JSON / PNG / CSV / PDF
   ├── report_generator.py  Explanation + reports
   └── facade.py            the public Explainer
```

The package consumes the **Phase 4 sample dict** (preprocessor output) and the
**Phase 5 model**. It never reads files directly.

## Design notes

* **Dependency injection** — every explainer takes the model (+ config, device)
  explicitly; the facade wires them together.
* **Robust attention extraction** — PyTorch's `TransformerEncoderLayer` calls
  `self_attn` with `need_weights=False` (fast SDPA path) and discards weights.
  Instead of monkeypatching module internals, `AttentionCapture` registers
  forward **pre-hooks** that record the exact inputs, then recomputes the
  attention with `need_weights=True`. This works across PyTorch versions.
* **GradCAM target discovery** — the last conv of a timm backbone often emits
  `1x1` maps; `find_last_spatial_conv` probes candidates and returns the deepest
  conv with genuine spatial extent.
* **Graceful degradation** — every explainer raises a typed
  `ExplainabilityError` subclass; the facade catches them and records a
  limitation in the report rather than crashing.
* **No hard SHAP dependency** — `shap` is optional; the framework ships a
  faithful KernelSHAP implementation.

## Adding a CAM method

Implement `CamMethod` in `interfaces.py` and register it in
`gradcam.compute_cam`:

```python
class MyCam(CamMethod):
    name = "mycam"
    def weights(self, activations, gradients):
        return ...  # [.., C] channel weights
```

## Adding a plot

Add a method to `Visualizer` and call it from `Explainer.visualize`.

## Tests

```bash
python -m pytest ai/explainability -q
```

47 tests (42 fast + 5 integration over the real STAM → preprocessing chain).
The integration tests build a tiny timm backbone at 32×32.

## Conventions

* Python 3.12+, PyTorch 2.x, PEP 8, type hints, docstrings, SOLID.
* Errors raise `ExplainabilityError` subclasses with stable `MXAI-<AREA>-<NNN>`
  codes.
* No TODOs / placeholders; new code ships with tests.
