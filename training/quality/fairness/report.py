"""Fairness report serialisation — JSON, CSV, HTML, PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluator import FairnessResult

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CropFusion Fairness Report</title>
<style>
  body { font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem auto;
         max-width: 960px; color: #1a202c; }
  h1 { font-size: 1.5rem; }
  .badge { display: inline-block; padding: .2rem .6rem; border-radius: 999px;
           color: #fff; font-size: .75rem; font-weight: 600; }
  .compliant { background: #38a169; } .at_risk { background: #d69e2e; }
  .violating { background: #e53e3e; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #e2e8f0; padding: .45rem .6rem; text-align: left;
           font-size: .85rem; }
  th { background: #f7fafc; }
</style>
</head>
<body>
<h1>CropFusion Fairness Report</h1>
<p>Task: <strong>{{ result.task }}</strong> &middot; Attribute:
   <strong>{{ result.attribute }}</strong></p>
<p>Overall: <span class="badge {{ result.overall_status }}">{{ result.overall_status }}</span></p>

<h2>Verdicts</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
  {% for v in result.verdicts %}
  <tr>
    <td>{{ v.metric }}</td><td>{{ "%.4f"|format(v.value) }}</td>
    <td>{{ "%.4f"|format(v.threshold) }}</td>
    <td><span class="badge {{ v.status }}">{{ v.status }}</span></td>
  </tr>
  {% endfor %}
</table>

<h2>Groups</h2>
<table>
  <tr><th>Group</th><th>Support</th><th>Accuracy</th><th>TPR</th><th>FPR</th>
      <th>Base rate</th><th>ECE</th><th>ROC-AUC</th></tr>
  {% for g in result.groups %}
  <tr>
    <td>{{ g.group }}</td><td>{{ g.support }}</td>
    <td>{{ "%.3f"|format(g.metrics.get("accuracy", 0)) }}</td>
    <td>{{ "%.3f"|format(g.metrics.get("tpr", 0)) }}</td>
    <td>{{ "%.3f"|format(g.metrics.get("fpr", 0)) }}</td>
    <td>{{ "%.3f"|format(g.metrics.get("base_rate", 0)) }}</td>
    <td>{{ "%.3f"|format(g.metrics.get("ece", 0)) }}</td>
    <td>{{ "%.3f"|format(g.metrics.get("roc_auc", 0)) }}</td>
  </tr>
  {% endfor %}
</table>
</body>
</html>
"""


class FairnessReportWriter:
    """Write a :class:`FairnessResult` in multiple formats."""

    def write(
        self,
        result: FairnessResult | dict[str, Any],
        out_dir: str | Path,
        *,
        formats: tuple[str, ...] = ("json", "csv", "html"),
    ) -> dict[str, Path]:
        """Persist the report and return ``{format: path}``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        written: dict[str, Path] = {}
        if "json" in formats:
            target = out / "fairness_report.json"
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written["json"] = target
        if "csv" in formats:
            target = out / "fairness_report.csv"
            rows = [
                {
                    "group": g["group"],
                    "support": g["support"],
                    **{f"metric_{k}": v for k, v in g.get("metrics", {}).items() if not isinstance(v, (dict, list))},
                }
                for g in payload.get("groups", [])
            ]
            pd.DataFrame(rows).to_csv(target, index=False)
            written["csv"] = target
        if "html" in formats:
            from jinja2 import Template

            target = out / "fairness_report.html"
            template = Template(_HTML_TEMPLATE, autoescape=True)
            target.write_text(template.render(result=payload), encoding="utf-8")
            written["html"] = target
        return written
