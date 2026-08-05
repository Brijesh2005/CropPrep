# CropFusion — Multimodal Explainable AI (MXAI) Framework (Phase 7)

A complete explainability framework for the Phase 5 multimodal model. It
answers **why a crop was recommended**, **why a yield was predicted**, **which
tabular features / image regions / observation dates / modalities mattered**,
and **how confident the model is** — for researchers, developers, agricultural
experts and farmers.

```
training/explainability/
├── __init__.py            # public API (Explainer + all explainers)
├── config.py              # ExplainabilityConfig (YAML + MXAI_* env)
├── interfaces.py          # ports: CamMethod / AttributionMethod / ConfidenceEstimator
├── utils.py               # AttentionCapture, GradCAM target discovery, batch/name helpers
├── shap_explainer.py      # self-contained KernelSHAP + SHAP plots + export
├── gradcam.py             # GradCAM / GradCAM++ / EigenCAM / LayerCAM for NDVI & EVI
├── integrated_gradients.py# tabular / image / shared-embedding integrated gradients
├── temporal_attention.py  # attention rollout + observation importance
├── cross_modal_attention.py # cross-attention + modality gates + token importance
├── uncertainty.py         # confidence, entropy, MC-dropout, ECE, reliability
├── counterfactual.py      # "what-if" perturbations (feature / image / temporal)
├── visualization.py       # SHAP / GradCAM / attention / confidence / calibration plots
├── exporter.py            # HTML / JSON / PNG / CSV / PDF
├── report_generator.py    # Explanation + farmer & research reports
├── facade.py              # the public Explainer (explain / report / visualize / export)
├── pyproject.toml
└── tests/                 # 47 tests
```

## Quick start

```python
from training.explainability import Explainer
from training.explainability.config import load_explainability_config

explainer = Explainer(
    model,                       # trained CropFusionModel
    preprocessor,                # fitted Phase 4 Preprocessor
    load_explainability_config("explainability.yaml"),
    observations=accepted,       # for SHAP background + historical comparison
    extractor=stam.get_patch,
)
explanation = explainer.explain(observation)

print(explanation.crop)                    # "Paddy"
print(explanation.yield_prediction)        # 6.12 (t/ha)
print(explanation.confidence)              # crop_conf / entropy / MC-dropout
print(explanation.top_features)            # [(feature, SHAP), ...]
print(explanation.temporal_ranking)        # most important observation dates
print(explanation.reasoning)               # farmer-friendly statements
```

## Public API

| Method | Purpose |
|--------|---------|
| `explain(observation, task="crop")` | full multimodal explanation |
| `explain_crop(observation)` | crop-recommendation explanation |
| `explain_yield(observation)` | yield-prediction explanation |
| `generate_report(observation, mode="farmer")` | farmer or research report |
| `visualize(explanation)` | render all figures |
| `export(explanation, formats=[...])` | HTML / JSON / PNG / CSV / PDF |

## What is explained

| Component | Explainer | Output |
|-----------|-----------|--------|
| Tabular features | SHAP (KernelSHAP) + integrated gradients | per-feature importance, force/waterfall/summary/bar/dependence/interaction plots |
| Image regions | GradCAM++ / GradCAM / EigenCAM / LayerCAM | per-timestep NDVI/EVI heatmaps, overlays, PNG/NumPy export |
| Temporal observations | attention rollout | per-date importance + ranking |
| Cross-modal | cross-attention + gates | image↔tabular attention, modality gates |
| Confidence | uncertainty estimator | confidence, entropy, MC-dropout, ECE, reliability diagram |
| Counterfactuals | what-if engine | "if rainfall +10% / NDVI -30% / drop date → would the crop change?" |
| Reports | report generator | farmer-friendly + research reports, historical comparison |

## Configuration

Everything is configurable via YAML (or `MXAI_*` env):

```python
from training.explainability.config import save_explainability_template
save_explainability_template("explainability.yaml")
```

See [docs/](docs/) for the developer, research, farmer, API and examples guides.
