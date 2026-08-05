# Examples

## 1. Explain a single field (full multimodal)

```python
from ai.explainability import Explainer
from ai.explainability.config import ExplainabilityConfig

explainer = Explainer(
    model,
    preprocessor,
    ExplainabilityConfig(
        shap={"background_size": 100, "max_samples": 256},
        uncertainty={"mc_dropout_samples": 20},
    ),
    observations=accepted,
    extractor=stam.get_patch,
)

explanation = explainer.explain(observation)
print(explanation.top_features)          # top SHAP features
print(explanation.temporal_ranking[:3])  # top observation dates
print(explanation.confidence)            # confidence / entropy / MC
figures = explainer.visualize(explanation)          # PNG figures
explainer.export(explanation, figures=figures)      # HTML/JSON/CSV/PNG/PDF
```

## 2. Farmer report

```python
report = explainer.generate_report(observation, mode="farmer")
print(report["why"])
```

## 3. Which modality drove the decision?

```python
print(explanation.gates)
# {'image_gate': 0.55, 'tabular_gate': 0.42, 'fusion_gate': 0.60}
explanation.cross_modal["cross_attention_score"]   # image→tabular attention
```

## 4. What-if (counterfactual)

```python
from ai.explainability import CounterfactualEngine

engine = CounterfactualEngine(model, feature_names=preprocessor.tabular.feature_names)
sample = explainer.sample(observation)
result = engine.explain(sample, perturbations=[
    {"feature": "rainfall_mm", "delta": 0.2, "mode": "multiply", "label": "rainfall +20%"},
    {"image": "ndvi", "factor": 0.6, "label": "NDVI -40%"},
    {"mask": 2, "label": "drop observation 2"},
])
for cf in result["counterfactuals"]:
    print(cf["label"], "-> crop changed:", cf["crop_changed"])
```

## 5. Calibration over a validation set

```python
import numpy as np
from ai.explainability import UncertaintyEstimator

estimator = UncertaintyEstimator(model)
confidences, correct = [], []
for obs in val_observations:
    sample = explainer.sample(obs)
    batch = {k: v.unsqueeze(0) for k, v in sample.items() if hasattr(v, "unsqueeze")}
    logits = model(batch).crop_logits
    probs = torch.softmax(logits, dim=-1)[0]
    confidences.append(float(probs.max()))
    correct.append(int(probs.argmax()) == int(sample["crop_label"].item()))
calib = estimator.calibration(np.asarray(confidences), np.asarray(correct), bins=10)
print("ECE:", calib["ece"])
```

## 6. Direct explainers (no preprocessor)

```python
from ai.explainability import SHAPExplainer, ImageExplainer, TemporalAttentionExplainer

shap = SHAPExplainer(model).explain(sample, background, kind="crop")   # local SHAP
cam  = ImageExplainer(model).explain(sample, index="ndvi", kind="crop")  # heatmaps
temporal = TemporalAttentionExplainer(model).explain(sample)             # importance
```
