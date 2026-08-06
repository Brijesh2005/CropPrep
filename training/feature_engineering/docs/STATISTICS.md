# Corpus Statistics & Balancing — R2.3

Two companions to feature building that answer "is this dataset trainable?"
before a single model runs: `CorpusStatistics` and `BalancingReport`, both in
`training/feature_engineering/`.

## CorpusStatistics

`CorpusStatistics.summarize(corpus)` computes one snapshot:

| Section | Contents |
| --- | --- |
| `total` / `status_counts` | accepted / rejected / error totals |
| `quality` | min / max / mean / median of accepted quality scores |
| `by_crop` / `by_year` / `by_season` / `by_location` | acceptance distributions per key |
| `yield_stats` | per-crop yield ranges + overall summary |
| `missing_labels` | how many accepted rows lack `crop` / `yield` |

It reads quality scores from the **resolved samples** (`ResolvedSample.quality_score`,
status `accepted`) — not from the observation objects — so the counts line up
with the corpus statuses.

## BalancingReport

`BalancingReport.from_corpus(corpus)` reports class balance for label surfaces
you can build a balanced dataset from:

| Report | Measures |
| --- | --- |
| `class_counts` | count + share per label class |
| `minority_majority_ratio` | ratio of smallest to largest class |
| `imbalance_ratio` | `majority / minority` (1.0 = perfectly balanced) |
| `balance_score` | normalised 0..1 summary (1.0 = balanced) |
| `threshold` | warning / severe imbalance thresholds |

`label_key` selects the surface: `"crop"` (default) or any other label column
present on the observation.

## Usage

```python
from training.feature_engineering import CorpusStatistics, BalancingReport

stats   = CorpusStatistics.summarize(corpus)
balance = BalancingReport.from_corpus(corpus, label_key="crop")

print(stats.to_dict()["missing_labels"])
print(balance.balance_score)          # 0.0..1.0
print(balance.imbalance_ratio)
```

`to_dict()` output is JSON-safe and designed to feed the QC report and the
migration record.

## Typical reading

- `imbalance_ratio > 10` (or `balance_score < ~0.35`) → warn: the dataset is
  crop-imbalanced; prefer class-weighted loss or oversampling in R3.
- High `missing_labels` → revisit the tabular table or the STAM resolver
  configuration before training.

See `training/feature_engineering/tests/test_statistics.py` and
`test_balancing.py` for the contract.
