"""Drift report serialisation — JSON, CSV, HTML and PDF.

All writers are pure functions on a :class:`~quality.drift.result.DriftReport`
and are exercised by the framework tests to guarantee every format is valid
and parseable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .result import DriftReport

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CropFusion Drift Report</title>
<style>
  body { font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem auto;
         max-width: 960px; color: #1a202c; }
  h1 { font-size: 1.5rem; }
  .badge { display: inline-block; padding: .2rem .6rem; border-radius: 999px;
           color: #fff; font-size: .75rem; font-weight: 600; }
  .low { background: #38a169; } .moderate { background: #d69e2e; }
  .high { background: #e53e3e; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #e2e8f0; padding: .45rem .6rem; text-align: left;
           font-size: .85rem; }
  th { background: #f7fafc; }
  .section { margin: 1.5rem 0; }
  .muted { color: #718096; font-size: .85rem; }
</style>
</head>
<body>
<h1>CropFusion Data Drift Report</h1>
<p class="muted">Generated: {{ summary.generated_at }}</p>
<p>Reference samples: <strong>{{ summary.reference_samples }}</strong> &middot;
   Current samples: <strong>{{ summary.current_samples }}</strong></p>
<p>Overall severity: <span class="badge {{ summary.overall_severity }}">{{ summary.overall_severity }}</span>
   &middot; Drift detected: <strong>{{ summary.drifted }}</strong></p>

<div class="section">
<h2>Feature drift ({{ summary.dimensions.features.total }})</h2>
<table>
  <tr><th>Feature</th><th>Type</th><th>Severity</th><th>Drifted</th>
      <th>PSI</th><th>JS</th><th>KS p-value</th></tr>
  {% for f in features %}
  <tr>
    <td>{{ f.feature }}</td><td>{{ f.dtype }}</td>
    <td><span class="badge {{ f.severity }}">{{ f.severity }}</span></td>
    <td>{{ f.drifted }}</td>
    <td>{{ "%.4f"|format(f.metrics.get("psi", 0)) }}</td>
    <td>{{ "%.4f"|format(f.metrics.get("js", 0)) }}</td>
    <td>{{ "%.4f"|format(f.metrics.get("ks_p_value", 1)) }}</td>
  </tr>
  {% endfor %}
</table>
</div>

{% if labels %}
<div class="section">
<h2>Label drift</h2>
<p>Severity: <span class="badge {{ labels.severity }}">{{ labels.severity }}</span> &middot;
   drifted: {{ labels.drifted }} &middot; novelty: {{ labels.novelty|length }}
   &middot; vanished: {{ labels.vanished|length }}</p>
<table>
  <tr><th>Category</th><th>Reference share</th><th>Current share</th><th>Shift</th></tr>
  {% for c in labels.categories[:20] %}
  <tr><td>{{ c.category }}</td><td>{{ "%.4f"|format(c.reference_share) }}</td>
      <td>{{ "%.4f"|format(c.current_share) }}</td>
      <td>{{ "%+.4f"|format(c.absolute_shift) }}</td></tr>
  {% endfor %}
</table>
</div>
{% endif %}

{% if predictions %}
<div class="section">
<h2>Prediction drift</h2>
<p>Severity: <span class="badge {{ predictions.severity }}">{{ predictions.severity }}</span> &middot;
   drifted: {{ predictions.drifted }} &middot; mode: {{ predictions.mode }}</p>
<p class="muted">Confidence shift: {{ "%.4f"|format(predictions.confidence_shift) }} &middot;
   entropy shift (bits): {{ "%.4f"|format(predictions.entropy_shift) }}</p>
</div>
{% endif %}

{% if spatial %}
<div class="section">
<h2>Spatial drift</h2>
<p>Severity: <span class="badge {{ spatial.severity }}">{{ spatial.severity }}</span> &middot;
   drifted: {{ spatial.drifted }}</p>
<p class="muted">Cells ref: {{ spatial.num_cells_reference }} &middot; current: {{ spatial.num_cells_current }}
   &middot; novel share: {{ "%.2f"|format(spatial.novel_cell_share) }}
   &middot; mean NN distance: {{ "%.1f"|format(spatial.mean_nearest_neighbour_km) }} km</p>
</div>
{% endif %}

{% if temporal %}
<div class="section">
<h2>Temporal drift</h2>
<p>Severity: <span class="badge {{ temporal.severity }}">{{ temporal.severity }}</span> &middot;
   drifted: {{ temporal.drifted }} &middot; trend: {{ temporal.trend_direction }}
   &middot; episodes: {{ temporal.episode_count }}</p>
</div>
{% endif %}

</body>
</html>
"""


class ReportWriter:
    """Write a :class:`DriftReport` in multiple formats."""

    def write(
        self, report: DriftReport, out_dir: str | Path, *, formats: tuple[str, ...] = ("json", "csv", "html", "pdf")
    ) -> dict[str, Path]:
        """Persist the report and return ``{format: path}``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        if "json" in formats:
            written["json"] = self.write_json(report, out / "drift_report.json")
        if "csv" in formats:
            written["csv"] = self.write_csv(report, out / "drift_report.csv")
        if "html" in formats:
            written["html"] = self.write_html(report, out / "drift_report.html")
        if "pdf" in formats:
            written["pdf"] = self.write_pdf(report, out / "drift_report.pdf")
        return written

    @staticmethod
    def write_json(report: DriftReport, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return target

    @staticmethod
    def write_csv(report: DriftReport, path: str | Path) -> Path:
        """Single CSV with a ``section`` column so all dimensions share one file."""
        target = Path(path)
        rows: list[dict[str, Any]] = []
        for feature in report.features:
            rows.append(
                {
                    "section": "feature",
                    "key": feature.feature,
                    "severity": feature.severity,
                    "drifted": feature.drifted,
                    "metric": "psi",
                    "value": feature.metrics.get("psi"),
                }
            )
        if report.labels:
            rows.append(
                {
                    "section": "label",
                    "key": "label",
                    "severity": report.labels.severity,
                    "drifted": report.labels.drifted,
                    "metric": "js",
                    "value": report.labels.metrics.get("js"),
                }
            )
        if report.predictions:
            rows.append(
                {
                    "section": "prediction",
                    "key": "prediction",
                    "severity": report.predictions.severity,
                    "drifted": report.predictions.drifted,
                    "metric": "js",
                    "value": report.predictions.metrics.get("js"),
                }
            )
        if report.spatial:
            rows.append(
                {
                    "section": "spatial",
                    "key": "spatial",
                    "severity": report.spatial.severity,
                    "drifted": report.spatial.drifted,
                    "metric": "novel_cell_share",
                    "value": report.spatial.novel_cell_share,
                }
            )
        if report.temporal:
            rows.append(
                {
                    "section": "temporal",
                    "key": "temporal",
                    "severity": report.temporal.severity,
                    "drifted": report.temporal.drifted,
                    "metric": "episode_count",
                    "value": report.temporal.episode_count,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(target, index=False)
        return target

    @staticmethod
    def write_html(report: DriftReport, path: str | Path) -> Path:
        from jinja2 import Template

        target = Path(path)
        template = Template(_HTML_TEMPLATE, autoescape=True)
        target.write_text(
            template.render(
                summary=report.summary(),
                features=report.features,
                labels=report.labels,
                predictions=report.predictions,
                spatial=report.spatial,
                temporal=report.temporal,
            ),
            encoding="utf-8",
        )
        return target

    @staticmethod
    def write_pdf(report: DriftReport, path: str | Path) -> Path:
        """Render the report to PDF with ReportLab."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        target = Path(path)
        styles = getSampleStyleSheet()
        accent = ParagraphStyle(
            "accent", parent=styles["Heading1"], textColor=colors.HexColor("#1a202c")
        )
        muted = ParagraphStyle(
            "muted", parent=styles["BodyText"], textColor=colors.HexColor("#718096"), fontSize=9
        )

        story: list[Any] = [Paragraph("CropFusion Data Drift Report", accent), Spacer(1, 4 * mm)]
        summary = report.summary()
        story.append(
            Paragraph(
                f"Generated {summary['generated_at']} &middot; ref={summary['reference_samples']} "
                f"current={summary['current_samples']} &middot; overall="
                f"<b>{summary['overall_severity']}</b> &middot; drifted={summary['drifted']}",
                muted,
            )
        )
        story.append(Spacer(1, 4 * mm))

        if report.features:
            story.append(Paragraph("Feature drift", styles["Heading2"]))
            header = ["Feature", "Type", "Severity", "Drifted", "PSI", "JS", "KS p"]
            data = [header]
            for f in report.features:
                data.append(
                    [
                        f.feature,
                        f.dtype,
                        f.severity,
                        str(f.drifted),
                        f"{f.metrics.get('psi', 0):.4f}",
                        f"{f.metrics.get('js', 0):.4f}",
                        f"{f.metrics.get('ks_p_value', 1):.4f}",
                    ]
                )
            table = Table(data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f7fafc")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.append(table)

        if report.labels:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph("Label drift", styles["Heading2"]))
            story.append(
                Paragraph(
                    f"Severity <b>{report.labels.severity}</b> &middot; drifted={report.labels.drifted} "
                    f"&middot; novelty={len(report.labels.novelty)} "
                    f"&middot; vanished={len(report.labels.vanished)}",
                    muted,
                )
            )

        doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm)
        doc.build(story)
        return target
