# Discussion

The experiments indicate that multimodal fusion does not consistently outperform the individual modalities under the current field-level labeling conditions.

All models remain close to the majority-chance balanced-accuracy level, so any
measured advantage is small in absolute terms. This is expected: black pepper
is intercropped under coconut, the two crops share physical pixels, and the
leading-window satellite composites (grid-cell averages, not per-crop
canopies) contain only weak per-crop discriminatory information. Location
features and shortcut probes (R5.10) show no suspicious >=65% signal, so the
weak results are a data-representation limit rather than a modelling failure.
