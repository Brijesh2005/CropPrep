"""Visualization + exporter tests."""

from __future__ import annotations

import numpy as np

from ai.explainability import Exporter, Visualizer
from ai.explainability.config import ExplainabilityConfig
from ai.explainability.report_generator import Explanation


def _explanation() -> Explanation:
    return Explanation(
        crop="Paddy",
        yield_prediction=6.12,
        confidence={"crop_conf": 0.8},
        feature_importance={"rainfall_mm": 0.4, "soil_moisture": -0.2},
        shap_values=np.asarray([0.4, -0.2, 0.1]),
        shap_base_value=0.0,
        temporal_importance={"d1": 0.5, "d2": 0.3},
        temporal_ranking=["d1", "d2"],
        gates={"image_gate": 0.5, "tabular_gate": 0.5},
    )


def test_visualizer_shap_plots(tmp_path):
    visualizer = Visualizer(tmp_path)
    names = ["rainfall", "soil", "temp"]
    values = np.asarray([0.4, -0.2, 0.1])
    paths = {
        "bar": visualizer.feature_importance_bar(values, names, tmp_path / "bar.png"),
        "waterfall": visualizer.shap_waterfall(values, 0.0, names, tmp_path / "wf.png"),
        "force": visualizer.shap_force(values, 0.0, names, tmp_path / "force.png"),
        "decision": visualizer.shap_decision(values, 0.0, names, tmp_path / "dec.png"),
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0


def test_visualizer_temporal_and_cross(tmp_path):
    visualizer = Visualizer(tmp_path)
    temporal = visualizer.temporal_timeline(
        np.asarray([0.5, 0.3, 0.2]), ["d1", "d2", "d3"], tmp_path / "timeline.png"
    )
    heatmap = visualizer.cross_modal_heatmap(
        np.asarray([[0.5, 0.1], [0.2, 0.3]]), ["d1", "d2"], ["a", "b"],
        tmp_path / "cross.png",
    )
    assert temporal.exists() and heatmap.exists()


def test_exporter_formats(tmp_path):
    exporter = Exporter(ExplainabilityConfig().export, output_dir=tmp_path / "exp")
    written = exporter.export(
        _explanation(),
        formats=["html", "json", "csv", "png", "pdf"],
        figures={},
    )
    assert written["html"].exists()
    assert written["json"].exists()
    assert written["csv"].exists()
    assert written["pdf"].exists()
    assert written["png"].is_dir()


def test_exporter_unknown_format(tmp_path):
    exporter = Exporter(ExplainabilityConfig().export, output_dir=tmp_path / "exp")
    try:
        exporter.export(_explanation(), formats=["docx"])
        raise AssertionError("expected ExportError")
    except Exception as exc:
        assert "unsupported export format" in str(exc)
