# Abstract

We evaluate whether jointly modeling satellite-derived vegetation composites
(rooted in Sentinel-2 indices) and environmental tabular descriptors improves
field-level classification of coconut versus black pepper in coastal Dakshina
Kannada, India, relative to either modality alone. Methods are governed by a
strict pre-registered protocol: a single balanced population built before
training, validation-only model selection, and a single frozen held-out test
evaluation. The test set is highly balanced (the minority class is never
discarded), and no evaluation methodology is altered to chase a nominal
threshold.

**Primary outcome (frozen test).** CropFusion balanced accuracy
0.4885, tabular
0.4962, imagery
0.4847; ROC-AUC
0.5337 vs tabular 0.4224 and
imagery 0.4566; macro-F1 0.2558
vs 0.4757 and 0.2449.

Multimodal fusion does not consistently outperform the individual modalities under the current field-level labeling conditions. The absolute
signal is weak (all models are near the majority-chance balanced-accuracy
level), which is consistent with the intercrop structure of black pepper under
coconut and the R5.8-R5.10 `no_signal` verdicts.
