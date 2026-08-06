# End-of-Run Reports

`generate_reports` turns a finished `TrainingResult` (and the resolved
`TrainingConfig`) into five artefacts. Everything comes from the in-memory
training history — nothing is re-read from disk inside the loop.

## Artefacts

| type | filename | content |
| --- | --- | --- |
| `training` | `training_report.md` | run name, epochs, steps, duration, early-stop status, best epoch / checkpoint, best-metrics table |
| `validation` | `validation_report.md` | per-epoch validation metrics table (`val_*`, `crop/*`, `yield/*`) |
| `metrics` | `metrics_report.md` | final-epoch classification (`crop/*`) and regression (`yield/*`) tables |
| `checkpoint` | `checkpoint_report.md` | checkpoint policy + the saved `*.pt` artifact list |
| `learning_curve` | `learning_curve.csv` | per-epoch scalar metrics (loss / accuracy / LR) for external plotting |

## Location

Reports go to `<general.output_dir>/reports` unless `general.reports_dir` is
set. `default_reports_dir(config)` resolves the location. `CropFusionTrainer`
writes the reports automatically at the end of `train()` when
`general.reports` is `true` (the default) and records the paths in
`result.reports`.

## Configuration

```yaml
general:
  reports: true          # generate at the end of train()
  reports_dir: null      # override: defaults to <output_dir>/reports
```

Env override: `TRN_GENERAL__REPORTS`, `TRN_GENERAL__REPORTS_DIR`.

## Usage

```python
from training.training.reports import generate_reports, default_reports_dir

paths = generate_reports(config, result)                 # default dir
paths = generate_reports(config, result, directory=dir)  # explicit dir

for report_type, path in paths.items():
    print(report_type, "->", path)
```

The CSV is scalars-only: non-numeric columns (e.g. the curriculum `stage`
string) are dropped, so it plots cleanly. When no validation loader was used,
the validation report notes `_No validation metrics recorded (no validation
loader)._` instead of failing.
