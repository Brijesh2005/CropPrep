# Research Guide

Use the research report and raw explanation components for publication-grade
analysis.

## The research report

```python
report = explainer.generate_report(observation, mode="research")
```

Contains:

* prediction (crop, class probabilities, scaled + physical yield),
* confidence / uncertainty (top-1 confidence, entropy, MC-dropout mean/std),
* per-feature SHAP values and the base value,
* integrated gradients on the tabular branch, image patches and the shared
  embedding,
* the adaptive-gate values (image / tabular / fusion) — which modality drove
  the decision,
* temporal importance (attention rollout, CLS → observation),
* per-timestep NDVI/EVI GradCAM heatmaps,
* cross-modal contribution heatmap `[timesteps × tabular features]`,
* counterfactual results, historical comparison, reasoning, limitations.

## Interpreting the components

| Component | Method | Reference |
|-----------|--------|-----------|
| Tabular SHAP | KernelSHAP | Lundberg & Lee, 2017 |
| Image CAM | GradCAM++ / EigenCAM / LayerCAM | Chattopadhay et al. 2018; Muhammad & Yeamin 2021; Jiang et al. 2021 |
| Temporal attention | attention rollout | Abnar & Zuidema, 2020 |
| Shared-embedding attribution | integrated gradients | Sundararajan et al., 2017 |
| Multi-task confidence | Monte-Carlo dropout | Gal & Ghahramani, 2016 |
| Calibration | ECE + reliability diagram | Guo et al., 2017 |
| Task balancing | GradNorm (training) | Chen et al., 2018 |

## Caveats to report

* SHAP reflects the model's learned associations, not causality.
* The cross-attention operates on pooled embeddings (single tokens), so the
  `cross_attention_score` is a scalar; the richer cross-modal signal is the
  `[T × F]` contribution heatmap composed from the temporal and tabular token
  attentions.
* The tabular feature-token attention is at **token-group** granularity (the
  TabTransformer pools all continuous features into one token); per-feature
  attribution comes from SHAP, not token attention.
* MC-dropout keeps BatchNorm in eval mode (single-sample safe).
