"""Explanation export: HTML, JSON, PNG, CSV and PDF.

:class:`Exporter` turns an :class:`~ai.training.explainability.report_generator.
Explanation` (plus optional rendered figures) into files on disk. PDF is built
with matplotlib's ``PdfPages`` so no extra dependency is required.
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import ExportConfig
from .exceptions import ExportError

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


class Exporter:
    """Export explanations to multiple formats."""

    def __init__(
        self,
        config: ExportConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or ExportConfig()
        self.output_dir = Path(output_dir or self.config.directory)

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def export(
        self,
        explanation: Any,
        *,
        formats: list[str] | None = None,
        figures: Mapping[str, Path] | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Export ``explanation`` to every requested format."""
        out_dir = Path(output_dir or self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figures = figures or {}
        requested = formats or list(self.config.formats)
        written: dict[str, Path] = {}
        for fmt in requested:
            fmt = fmt.lower()
            if fmt == "html":
                written["html"] = self.export_html(explanation, out_dir / "explanation.html", figures)
            elif fmt == "json":
                written["json"] = self.export_json(explanation, out_dir / "explanation.json")
            elif fmt == "csv":
                written["csv"] = self.export_csv(explanation, out_dir / "features.csv")
            elif fmt == "png":
                written["png"] = self.export_png(figures, out_dir / "figures")
            elif fmt == "pdf":
                written["pdf"] = self.export_pdf(explanation, figures, out_dir / "explanation.pdf")
            else:
                raise ExportError(f"unsupported export format {fmt!r}")
        return written

    # ------------------------------------------------------------------ #
    # Individual formats
    # ------------------------------------------------------------------ #

    def export_json(self, explanation: Any, path: str | Path) -> Path:
        data = explanation.to_dict() if hasattr(explanation, "to_dict") else dict(explanation)
        out = Path(path)
        out.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return out

    def export_csv(self, explanation: Any, path: str | Path) -> Path:
        out = Path(path)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", "importance"])
            importance = getattr(explanation, "feature_importance", None) or {}
            for name, value in importance.items():
                writer.writerow([name, f"{value:.6f}"])
        return out

    def export_png(self, figures: Mapping[str, Path], directory: str | Path) -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, path in figures.items():
            target = out_dir / f"{name}.png"
            if Path(path).exists():
                target.write_bytes(Path(path).read_bytes())
            else:
                raise ExportError(f"figure {name!r} not found at {path}")
        return out_dir

    def export_html(
        self,
        explanation: Any,
        path: str | Path,
        figures: Mapping[str, Path] | None = None,
    ) -> Path:
        figures = figures or {}
        image_tags = "".join(
            f"<h3>{name.replace('_', ' ').title()}</h3>"
            f"<img src='data:image/png;base64,{_b64(Path(p))}' style='max-width:100%'/>"
            for name, p in figures.items() if Path(p).exists()
        )
        html = _render_html(explanation, image_tags)
        out = Path(path)
        out.write_text(html, encoding="utf-8")
        return out

    def export_pdf(
        self,
        explanation: Any,
        figures: Mapping[str, Path] | None,
        path: str | Path,
    ) -> Path:
        """Assemble a PDF from the explanation text and rendered figures."""
        if plt is None:
            raise ExportError("matplotlib is required for PDF export")
        figures = figures or {}
        from matplotlib.backends.backend_pdf import PdfPages

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out) as pdf:
            # Text page(s).
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.95, _plain_report(explanation), fontsize=8, va="top",
                     family="monospace", wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            # Figure pages.
            for name, fpath in figures.items():
                if not Path(fpath).exists():
                    continue
                import matplotlib.image as mpimg

                img = mpimg.imread(str(fpath))
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(name.replace("_", " ").title(), fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
        return out


# --------------------------------------------------------------------------- #
# HTML / text rendering
# --------------------------------------------------------------------------- #


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _render_html(explanation: Any, image_tags: str) -> str:
    title = getattr(explanation, "crop", "CropFusion") or "CropFusion"
    yield_val = getattr(explanation, "yield_prediction", None)
    confidence = getattr(explanation, "confidence", {}) or {}
    reasoning = getattr(explanation, "reasoning", []) or []
    limitations = getattr(explanation, "limitations", []) or []

    rows: list[str] = []
    if yield_val is not None:
        rows.append(f"<p><b>Recommended crop:</b> {title}</p>")
        rows.append(f"<p><b>Expected yield:</b> {yield_val:.2f} t/ha</p>")
        rows.append(f"<p><b>Confidence:</b> {confidence.get('crop_conf', 0) * 100:.1f}%</p>")
    if reasoning:
        rows.append("<h3>Reasoning</h3><ul>" + "".join(f"<li>{r}</li>" for r in reasoning) + "</ul>")
    if limitations:
        rows.append("<h3>Limitations</h3><ul>" + "".join(f"<li>{l}</li>" for l in limitations) + "</ul>")

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>CropFusion Explanation</title></head>
<body style='font-family:system-ui,sans-serif;margin:2rem'>
<h1>CropFusion Explanation</h1>
{''.join(rows)}
<h2>Visualizations</h2>
{image_tags}
</body></html>"""


def _plain_report(explanation: Any) -> str:
    lines = ["CropFusion Explanation", "=" * 40]
    crop = getattr(explanation, "crop", None)
    if crop:
        lines.append(f"Recommended crop: {crop}")
    yield_val = getattr(explanation, "yield_prediction", None)
    if yield_val is not None:
        lines.append(f"Expected yield:   {yield_val:.2f} t/ha")
    confidence = getattr(explanation, "confidence", {}) or {}
    if confidence.get("crop_conf") is not None:
        lines.append(f"Confidence:       {confidence['crop_conf'] * 100:.1f}%")
    lines.append("")
    lines.append("Reasoning")
    lines.append("-" * 40)
    for r in (getattr(explanation, "reasoning", []) or []):
        lines.append(f"* {r}")
    lines.append("")
    lines.append("Limitations")
    lines.append("-" * 40)
    for l in (getattr(explanation, "limitations", []) or []):
        lines.append(f"* {l}")
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch_is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def torch_is_tensor(value: Any) -> bool:
    try:
        import torch

        return isinstance(value, torch.Tensor)
    except Exception:
        return False
