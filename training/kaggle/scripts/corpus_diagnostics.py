"""Corpus rejection diagnostics — root-cause the accepted/rejected/error split.

Loads a ``corpus.json`` written by the ObservationResolver (run_pipeline) and
produces a machine-readable rejection breakdown: which quality gates fired,
how they co-occur, and how rejections distribute over years / seasons /
locations. Includes representative sample cells per rejection bucket so the
root cause can be audited without replaying the full resolution.

Run::

    python training/kaggle/scripts/corpus_diagnostics.py \\
        --corpus kaggle_runs/train-dk-bridge/reports/CropPrep/training/kaggle/outputs/reports/corpus.json \\
        --output training/artifacts/corpus_diagnostics

Exit code is 0 on success; the JSON report is always written even when the
corpus is empty or 100% rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    """Force the repository root to the front of ``sys.path``."""
    repo_root = repo_root.resolve()
    root = str(repo_root)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    repo_training = (repo_root / "training").resolve()
    for entry in list(sys.path):
        if entry == root or entry == "":
            continue
        shadow = Path(entry) / "training"
        if shadow.exists() and shadow.resolve() != repo_training:
            print(f"[corpus_diagnostics] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)


def _load_corpus(path: Path) -> tuple[list[dict], dict]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return raw.get("samples", []), raw.get("config", {})


def _sample_brief(sample: dict) -> dict:
    obs = sample.get("observation") or {}
    tabular = obs.get("tabular") or {}
    sequence = obs.get("sequence") or {}
    temporal = obs.get("temporal") or {}
    return {
        "location_id": sample.get("location_id"),
        "name": sample.get("name"),
        "lon": sample.get("lon"),
        "lat": sample.get("lat"),
        "year": sample.get("year"),
        "season": sample.get("season"),
        "quality_score": sample.get("quality_score"),
        "error": sample.get("error"),
        "matched_level": tabular.get("matched_level"),
        "tabular_source": str(tabular.get("source_path", "")).split("/")[-1] or None,
        "crop": obs.get("crop"),
        "yield_value": obs.get("yield_value"),
        "image_dates": temporal.get("observation_dates"),
        "ndvi_count": (obs.get("provenance") or {}).get("ndvi_count"),
        "evi_count": (obs.get("provenance") or {}).get("evi_count"),
        "paired_count": (obs.get("provenance") or {}).get("paired_count"),
        "observation_id": obs.get("observation_id"),
    }


def analyze(samples: list[dict]) -> dict:
    counts = {"accepted": 0, "rejected": 0, "error": 0}
    code_counter = Counter()
    combo_counter = Counter()
    for sample in samples:
        counts[sample["status"]] += 1
        issues = (sample.get("observation") or {}).get("quality", {}).get("issues", [])
        codes = tuple(sorted(i["code"] for i in issues))
        combo_counter[codes] += 1
        for issue in issues:
            code_counter[issue["code"]] += 1

    code_breakdown = []
    for code, total in code_counter.most_common():
        by_status = Counter(s["status"] for s in samples if any(
            i["code"] == code for i in (s.get("observation") or {}).get("quality", {}).get("issues", [])
        ))
        code_breakdown.append({
            "code": code,
            "count": total,
            "accepted": by_status.get("accepted", 0),
            "rejected": by_status.get("rejected", 0),
        })

    combo_breakdown = []
    for combo, total in combo_counter.most_common():
        members = [s for s in samples if tuple(sorted(
            i["code"] for i in (s.get("observation") or {}).get("quality", {}).get("issues", [])
        )) == combo]
        statuses = Counter(s["status"] for s in members)
        combo_breakdown.append({
            "issues": list(combo),
            "count": total,
            "accepted": statuses.get("accepted", 0),
            "rejected": statuses.get("rejected", 0),
            "error": statuses.get("error", 0),
            "representative": [_sample_brief(s) for s in members[:3]],
        })

    total = len(samples) or 1
    return {
        "summary": {
            "total": len(samples),
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "errors": counts["error"],
            "acceptance_rate": round(counts["accepted"] / total, 6),
        },
        "issue_breakdown": code_breakdown,
        "combo_breakdown": combo_breakdown,
    }


def _distributions(samples: list[dict]) -> dict:
    rejected = [s for s in samples if s["status"] == "rejected"]
    accepted = [s for s in samples if s["status"] == "accepted"]
    out: dict[str, dict] = {}

    def _counter(rows, key):
        return dict(sorted(Counter(row.get(key) for row in rows).items(), key=lambda kv: str(kv[0])))

    out["rejected_by_year"] = _counter(rejected, "year")
    out["rejected_by_season"] = _counter(rejected, "season")
    out["rejected_by_location"] = _counter(rejected, "name")
    out["accepted_by_year"] = _counter(accepted, "year")
    out["accepted_by_season"] = _counter(accepted, "season")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-corpus-diagnostics")
    parser.add_argument("--corpus", required=True, help="Path to corpus.json")
    parser.add_argument("--output", default=None, help="Output dir for the report JSON")
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        print(f"[corpus_diagnostics] corpus not found: {corpus_path}", file=sys.stderr)
        return 2

    samples, config = _load_corpus(corpus_path)
    report = {
        "corpus": str(corpus_path),
        "created_at": config.get("created_at"),
        "config": config,
        **analyze(samples),
        "distributions": _distributions(samples),
    }

    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        target = output / "corpus_rejection_report.json"
        target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[corpus_diagnostics] wrote {len(json.dumps(report))} bytes -> {target}")

    summary = report["summary"]
    print(f"[corpus_diagnostics] total={summary['total']} "
          f"accepted={summary['accepted']} rejected={summary['rejected']} "
          f"errors={summary['errors']} rate={summary['acceptance_rate']}")
    for entry in report["combo_breakdown"]:
        print(f"  {entry['count']:>7}  issues={entry['issues']} "
              f"(acc={entry['accepted']} rej={entry['rejected']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
