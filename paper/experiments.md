# Experiments

- **E-1**: Tabular-only (no location) — headline environmental model.
- **E-2**: Tabular-with-location — documents the geographic-prior
  sensitivity explicitly.
- **E-3**: Imagery-only — temporal satellite composite sequence.
- **E-4**: CropFusion — tabular + imagery, fusion mechanism chosen on
  validation.
- **E-5**: Ablations — tabular vs imagery vs fusion; fusion mechanisms
  (concat / gated / cross-attention), all validation-based.
- **E-6**: Robustness — the `min-observation >= 3` rule applied identically
  to all three models.
- **E-7**: Stratified bootstrap confidence intervals (seed 42) on the frozen
  test metrics.

Artifacts: `reports/final/*` (results, ablation, CI, confusion matrices,
provenance, figures, tables). Reproducibility hashes are in
`reports/final/final_provenance.json`.
