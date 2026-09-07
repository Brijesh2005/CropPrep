# Limitations

- Raw Sentinel-2 patches are only available on the Kaggle platform; this
  experiment's imagery modality uses the matched DK grid vegetation-impact
  composites (the strongest locally reproducible representation). An
  EfficientNet-style patch model (the original CropFusion image encoder) is
  therefore out of scope here.
- The experiment is CPU-only; neural models are deliberately small.
- Pepper is the minority class: 196 pepper fields in the test set limits
  statistical power (reported via bootstrap CIs).
- Field grouping is by survey-ID within a taluk/hobli/village; satellite
  composites are cell averages, so sub-pixel intercropping cannot be resolved.
- Crop_Extent unit is UNKNOWN; only its relative within-field ordering built
  the R5.9 target.
