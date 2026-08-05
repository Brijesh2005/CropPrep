# API Documentation

## `Explainer` (facade)

```python
Explainer(model, preprocessor=None, config=None, observations=None,
          extractor=None, feature_names=None, crop_classes=None)
```

| Method | Returns |
|--------|---------|
| `explain(observation, task="crop")` | `Explanation` |
| `explain_crop(observation)` | `Explanation` |
| `explain_yield(observation)` | `Explanation` |
| `predict(observation)` | raw logits / scaled yield / gates |
| `generate_report(observation, mode="farmer"\|"research")` | `dict` |
| `visualize(explanation, output_dir=None)` | `{name: Path}` |
| `export(explanation, formats=None, figures=None)` | `{format: Path}` |

## `Explanation`

`crop`, `crop_probs`, `yield_prediction` (physical t/ha), `confidence`,
`feature_importance`, `shap_values`, `shap_base_value`, `image_heatmaps`,
`image_overlays`, `temporal_importance`, `temporal_ranking`, `cross_modal`,
`counterfactuals`, `gates`, `integrated_gradients`, `reasoning`,
`limitations`, `historical`, `raw`, plus `top_features` and `to_dict()`.

## Explainer classes

* `SHAPExplainer(model, config, device)` — `explain`, `global_importance`,
  `kernel_shap`, `gradient_shap`, `to_csv`, `to_json`.
* `ImageExplainer(model, config, device)` — `explain`, `overlay`,
  `export_png`, `export_numpy`.
* `TemporalAttentionExplainer(model, config, device)` — `explain`,
  `head_attention`.
* `CrossModalExplainer(model, config, device)` — `explain`.
* `UncertaintyEstimator(model, config, device)` — `crop_confidence`,
  `entropy`, `mc_dropout`, `yield_confidence`, `calibration`,
  `confidence_distribution`.
* `CounterfactualEngine(model, config, device, feature_names)` — `explain`,
  `perturb_tabular`, `perturb_image`, `mask_observation`, `predict`.
* `TabularIntegratedGradients` / `ImageIntegratedGradients` /
  `SharedEmbeddingIntegratedGradients` — `attribute`.
* `Visualizer(output_dir, ...)` — all plot methods.
* `Exporter(config, output_dir)` — `export`, `export_html/json/csv/png/pdf`.

## Configuration

`ExplainabilityConfig` sections: `general`, `shap`, `cam`,
`temporal_attention`, `cross_modal`, `integrated_gradients`,
`counterfactual`, `uncertainty`, `report`, `visualization`, `export`.
Load via `load_explainability_config(path)`; override via `MXAI_<SECTION>__<KEY>`
env vars; template via `save_explainability_template(path)`.
