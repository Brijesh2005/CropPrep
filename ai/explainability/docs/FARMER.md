# Farmer Guide

This guide explains how the CropFusion explainability framework produces the
plain-language explanations shown to farmers and agricultural experts.

## What you see

For every field, the framework produces:

```
Recommended crop:  Paddy
Expected yield:    6.12 t/ha
Confidence:        94.6%

Why?
• The rainfall pattern was positively influential for this prediction.
• Seasonal vegetation remained healthy.
• The most influential observation was 2020-09-01.
• The predicted yield is higher than the historical average for this crop.

What if...?
• If rainfall increased by 10%  → the recommendation would not change.
• If NDVI dropped 30%           → the recommendation would change to Wheat.
```

## How to get it

```python
from ai.explainability import Explainer

explainer = Explainer(model, preprocessor, config,
                      observations=accepted, extractor=stam.get_patch)
report = explainer.generate_report(observation, mode="farmer")
# report["why"]            -> the plain-language reasoning list
# report["top_factors"]    -> the features that mattered most
# report["important_dates"]-> observation dates that mattered most
# report["confidence_percent"]
# report["limitations"]
```

Export it:

```python
explanation = explainer.explain(observation)
explainer.export(explanation, formats=["html", "pdf", "json"])
```

A self-contained `explanation.html` (or `explanation.pdf`) is written with the
reasoning, the important dates, the top factors and the GradCAM maps overlaid
on the satellite patches.

## Reading the outputs

* **Confidence** is the model's top-1 class probability (0–100%). Low
  confidence (< 60%) is flagged in the limitations.
* **Top factors** are the tabular fields (e.g. rainfall, soil moisture) whose
  change would move the prediction the most.
* **Important dates** are the observation dates the temporal attention focused
  on (vegetation state on those dates mattered most).
* **What-if** rows show whether a plausible change (rainfall up, NDVI down,
  dropping a bad date) would flip the crop recommendation.
* **Historical comparison** compares the predicted yield with the average yield
  previously observed for the same crop in the region.

## Limitations you should know

* The explanation reflects what the model *learned*, not a field experiment —
  correlation is not causation.
* GradCAM highlights the regions of the satellite patch the model focused on;
  a low value does not mean the vegetation there was bad.
* When imagery is unavailable, only the tabular explanation is produced.
