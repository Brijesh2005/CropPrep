"""Self-contained HTML performance/quality dashboard.

Renders a single ``.html`` file (no external assets) combining:

* runtime metrics (``MetricsRegistry.snapshot()`` from the backend),
* model / inference status,
* the latest drift report (``drift_report.json``),
* the latest fairness report (``fairness_report.json``),
* optimization benchmarks (``benchmark_report.json``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_CSS = """
  body { font-family: -apple-system,'Segoe UI',sans-serif; margin: 2rem auto; max-width: 980px;
         color: #1a202c; background: #fafafa; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 1.6rem; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: .9rem 1.1rem; margin: .8rem 0; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; }
  .kpi { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: .8rem; text-align: center; }
  .kpi .value { font-size: 1.5rem; font-weight: 700; }
  .kpi .label { color: #718096; font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; }
  .badge { display:inline-block; padding:.15rem .55rem; border-radius:999px; color:#fff; font-size:.72rem; font-weight:600; }
  .low,.compliant { background:#38a169; } .moderate,.at_risk { background:#d69e2e; }
  .high,.violating { background:#e53e3e; }
  table { border-collapse: collapse; width: 100%; margin: .6rem 0; }
  th, td { border: 1px solid #e2e8f0; padding: .4rem .55rem; text-align: left; font-size: .83rem; }
  th { background: #f7fafc; }
  .muted { color: #718096; font-size: .8rem; }
"""


def render_performance_dashboard(
    metrics_snapshot: Mapping[str, Any] | None = None,
    *,
    drift_report_path: str | Path | None = None,
    fairness_report_path: str | Path | None = None,
    benchmark_report_path: str | Path | None = None,
    model_status: Mapping[str, Any] | None = None,
) -> str:
    """Return the full HTML for the dashboard."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = [f"<html><head><meta charset='utf-8'><title>CropFusion Performance Dashboard</title>",
             f"<style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>CropFusion Performance Dashboard</h1><p class='muted'>Generated {now} UTC</p>")

    parts.append(_render_system(metrics_snapshot, model_status))
    if drift_report_path and Path(drift_report_path).exists():
        parts.append(_render_drift(_read(drift_report_path)))
    if fairness_report_path and Path(fairness_report_path).exists():
        parts.append(_render_fairness(_read(fairness_report_path)))
    if benchmark_report_path and Path(benchmark_report_path).exists():
        parts.append(_render_benchmark(_read(benchmark_report_path)))

    parts.append("</body></html>")
    return "\n".join(parts)


def _render_system(snapshot: Mapping[str, Any] | None, model: Mapping[str, Any] | None) -> str:
    snapshot = dict(snapshot or {})
    by_path = snapshot.get("by_path", {})
    model = dict(model or {})
    ready = model.get("ready")
    ready_cls = "low" if ready else "high"
    ready_label = "ready" if ready else "not-ready"
    ready_badge = f"<span class='badge {ready_cls}'>{ready_label}</span>"

    rows = "".join(
        f"<tr><td>{path}</td><td>{entry['requests']}</td><td>{entry.get('avg_ms', 0):.1f} ms</td>"
        f"<td>{entry.get('errors', 0)}</td></tr>"
        for path, entry in sorted(by_path.items())
    )
    return f"""
    <div class='card'>
      <h2>System status</h2>
      <div class='grid'>
        {_kpi('Requests', snapshot.get('requests', 0))}
        {_kpi('Requests/sec', snapshot.get('requests_per_second', 0))}
        {_kpi('Avg latency (ms)', snapshot.get('avg_latency_ms', 0))}
        {_kpi('Errors', snapshot.get('errors', 0))}
        {_kpi('Uptime (s)', snapshot.get('uptime_seconds', 0))}
        {_kpi('Model', ready_badge)}
      </div>
      {_kpi_table('Per-path latency', rows)}
    </div>"""


def _render_drift(data: Mapping[str, Any]) -> str:
    dims = data.get("dimensions", {})
    features = dims.get("features", {})
    label = dims.get("label") or {}
    spatial = dims.get("spatial") or {}
    temporal = dims.get("temporal") or {}
    return f"""
    <div class='card'>
      <h2>Data drift <span class='badge {data.get('overall_severity', 'low')}'>{data.get('overall_severity')}</span></h2>
      <p class='muted'>drifted={data.get('drifted')} &middot; ref={data.get('reference_samples')}
         current={data.get('current_samples')}</p>
      <div class='grid'>
        {_kpi('Features drifted', features.get('drifted', 0))}
        {_kpi('High severity', features.get('high', 0))}
        {_kpi('Label', label.get('severity', 'n/a'))}
        {_kpi('Spatial', spatial.get('severity', 'n/a'))}
        {_kpi('Temporal', temporal.get('severity', 'n/a'))}
      </div>
    </div>"""


def _render_fairness(data: Mapping[str, Any]) -> str:
    verdict_rows = "".join(
        f"<tr><td>{v['metric']}</td><td>{v['value']:.4f}</td><td>{v['threshold']:.4f}</td>"
        f"<td><span class='badge {v['status']}'>{v['status']}</span></td></tr>"
        for v in data.get("verdicts", [])
    )
    group_rows = "".join(
        f"<tr><td>{g['group']}</td><td>{g.get('support', 0)}</td>"
        f"<td>{g.get('metrics', {}).get('accuracy', 0):.3f}</td>"
        f"<td>{g.get('metrics', {}).get('tpr', 0):.3f}</td>"
        f"<td>{g.get('metrics', {}).get('fpr', 0):.3f}</td></tr>"
        for g in data.get("groups", [])
    )
    return f"""
    <div class='card'>
      <h2>Fairness <span class='badge {data.get('overall_status', 'at_risk')}'>{data.get('overall_status')}</span></h2>
      <p class='muted'>task={data.get('task')} &middot; attribute={data.get('attribute')}</p>
      {_kpi_table('Parity verdicts', verdict_rows)}
      {_kpi_table('Groups', group_rows)}
    </div>"""


def _render_benchmark(data: Mapping[str, Any]) -> str:
    rows = []
    for variant in data.get("variants", []):
        latency = variant.get("latency", {})
        rows.append(
            f"<tr><td>{variant['name']}</td>"
            f"<td>{latency.get('mean_ms', 0):.2f}</td>"
            f"<td>{latency.get('p95_ms', 0):.2f}</td>"
            f"<td>{latency.get('p99_ms', 0):.2f}</td>"
            f"<td>{variant.get('throughput', {}).get('samples_per_second', 0):.1f}</td></tr>"
        )
    return f"""
    <div class='card'>
      <h2>Optimization benchmark</h2>
      <p class='muted'>model={data.get('model', '')} &middot; device={data.get('device', '')}</p>
      {_kpi_table('Variants (ms)', ''.join(rows))}
    </div>"""


def _kpi(label: str, value: Any) -> str:
    return f"<div class='kpi'><div class='value'>{value}</div><div class='label'>{label}</div></div>"


def _kpi_table(title: str, rows: str) -> str:
    return f"<h3 class='muted'>{title}</h3><table><tbody>{rows or '<tr><td>no data</td></tr>'}</tbody></table>"


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
