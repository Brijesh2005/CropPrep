"""Visualization suite for explainability.

Renders SHAP plots (bar / summary / waterfall / force / decision / dependence
/ interaction), GradCAM overlays, attention heatmaps, a temporal timeline, the
cross-modal heatmap, confidence distribution and the calibration (reliability)
curve. Every figure is saved as a PNG.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional
    plt = None

from .exceptions import VisualizationError


class Visualizer:
    """Renders explanation figures into ``output_dir``."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dpi: int = 110,
        colormap: str = "jet",
        max_features: int = 15,
    ) -> None:
        if plt is None:
            raise VisualizationError("matplotlib is required for visualization")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.colormap = colormap
        self.max_features = max_features

    # ------------------------------------------------------------------ #
    # SHAP plots
    # ------------------------------------------------------------------ #

    def feature_importance_bar(
        self, values: np.ndarray, feature_names: Sequence[str], path: str | Path
    ) -> Path:
        """Global / mean |SHAP| bar plot."""
        order = np.argsort(-np.abs(values))
        k = min(len(values), self.max_features)
        names = [str(feature_names[i]) for i in order[:k]]
        vals = [float(np.abs(values[i])) for i in order[:k]]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * k)))
        ax.barh(range(k)[::-1], vals, color="#2c7fb8")
        ax.set_yticks(range(k)[::-1])
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title("Feature Importance (mean |SHAP|)")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def shap_summary(
        self,
        shap_matrix: np.ndarray,
        feature_values: np.ndarray | None,
        feature_names: Sequence[str],
        path: str | Path,
    ) -> Path:
        """Beeswarm summary plot."""
        fig, ax = plt.subplots(figsize=(8, 0.5 * shap_matrix.shape[1] + 2))
        order = np.argsort(-np.abs(shap_matrix).mean(axis=0))
        for rank, f in enumerate(order):
            values = shap_matrix[:, f]
            if feature_values is not None:
                colors = feature_values[:, f]
                colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-9)
            else:
                colors = np.zeros_like(values)
            x = values + np.random.RandomState(42).uniform(-0.01, 0.01, size=values.size)
            ax.scatter(x, np.full(values.size, rank), c=colors, s=4, alpha=0.6, cmap="coolwarm")
            ax.axhline(rank - 0.5, color="gray", lw=0.4, alpha=0.3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([str(feature_names[i]) for i in order], fontsize=8)
        ax.set_xlabel("SHAP value")
        ax.set_title("SHAP Summary (color = feature value)")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def shap_waterfall(
        self, values: np.ndarray, base_value: float, feature_names: Sequence[str],
        path: str | Path,
    ) -> Path:
        """Waterfall plot from the base value to f(x)."""
        k = min(len(values), self.max_features)
        order = np.argsort(-np.abs(values))[:k]
        cumulative = base_value
        steps = []
        for i in reversed(order):
            steps.append(cumulative)
            cumulative += values[i]
        steps.append(cumulative)
        names = [str(feature_names[i]) for i in reversed(order)] + ["f(x)"]

        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * k)))
        pos = list(range(k + 1))
        ax.plot(pos, steps, "o-", color="#444", lw=0.8)
        for i, step in enumerate(steps):
            ax.plot([i, i], [min(steps), step], color="#9ecae1", lw=4, alpha=0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("prediction")
        ax.set_title(f"SHAP Waterfall (base = {base_value:.3f})")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def shap_force(
        self, values: np.ndarray, base_value: float, feature_names: Sequence[str],
        path: str | Path,
    ) -> Path:
        """Force-style horizontal bar showing positive / negative pushes."""
        order = np.argsort(-np.abs(values))[: self.max_features]
        names = [str(feature_names[i]) for i in order]
        vals = [float(values[i]) for i in order]

        fig, ax = plt.subplots(figsize=(9, 2.6))
        cumulative = base_value
        for i, value in enumerate(vals):
            left = min(cumulative, cumulative + value)
            width = abs(value)
            ax.barh(0, width, left=left, color="#d73027" if value > 0 else "#4575b4", height=0.7)
            ax.text(cumulative + value, 0.45, names[i], fontsize=6, ha="center", rotation=0)
            cumulative += value
        ax.axvline(base_value, color="black", ls="--", lw=0.8)
        ax.set_xlim(min(min(vals + [base_value]), 0) - 0.5, cumulative + 0.5)
        ax.set_yticks([])
        ax.set_title("SHAP Force (red pushes up, blue pushes down)")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def shap_decision(
        self, values: np.ndarray, base_value: float, feature_names: Sequence[str],
        path: str | Path,
    ) -> Path:
        """Decision (cumulative contribution) plot."""
        order = np.argsort(-np.abs(values))
        cumulative = np.cumsum(values[order])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([base_value] + [base_value + c for c in cumulative], range(len(values) + 1), "-o", ms=3)
        ax.axvline(base_value, color="gray", ls="--", lw=0.8)
        ax.set_yticks(range(len(values) + 1))
        ax.set_yticklabels(["base"] + [str(feature_names[i]) for i in order], fontsize=7)
        ax.set_xlabel("cumulative SHAP contribution")
        ax.set_title("SHAP Decision Plot")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def shap_dependence(
        self, values: np.ndarray, feature_values: np.ndarray, feature_index: int,
        feature_names: Sequence[str], path: str | Path,
    ) -> Path:
        """SHAP value vs feature value for one feature."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(feature_values[:, feature_index], values[:, feature_index], s=8, alpha=0.6)
        ax.set_xlabel(str(feature_names[feature_index]))
        ax.set_ylabel(f"SHAP value for {feature_names[feature_index]}")
        ax.set_title("SHAP Dependence Plot")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def shap_interaction(
        self, interaction_matrix: np.ndarray, feature_names: Sequence[str], path: str | Path
    ) -> Path:
        """SHAP interaction heatmap (mean |interaction| over samples)."""
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(np.abs(interaction_matrix), cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(feature_names, fontsize=7)
        ax.set_title("SHAP Interaction (mean |interaction|)")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    # ------------------------------------------------------------------ #
    # Image / attention / temporal
    # ------------------------------------------------------------------ #

    def gradcam_overlay(self, overlay_rgb: np.ndarray, path: str | Path) -> Path:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.clip(overlay_rgb, 0, 1))
        ax.axis("off")
        ax.set_title("GradCAM Overlay")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def attention_heatmap(
        self, attention: np.ndarray, tick_labels: Sequence[str] | None,
        path: str | Path, title: str = "Attention Heatmap",
    ) -> Path:
        """Heatmap of a (possibly per-layer) attention matrix."""
        matrix = np.asarray(attention)
        if matrix.ndim == 3:  # [H, T, T] -> average heads
            matrix = matrix.mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="viridis")
        fig.colorbar(im, ax=ax)
        if tick_labels is not None and len(tick_labels) == matrix.shape[0]:
            ax.set_xticks(range(len(tick_labels)))
            ax.set_yticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def temporal_timeline(
        self, importance: np.ndarray, labels: Sequence[str] | None, path: str | Path
    ) -> Path:
        """Bar / line plot of per-observation temporal importance."""
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(importance))
        ax.bar(x, importance, color="#66c2a5")
        ax.set_xlabel("observation date")
        ax.set_ylabel("temporal importance")
        ax.set_title("Temporal Observation Importance")
        if labels is not None and len(labels) == len(importance):
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def cross_modal_heatmap(
        self, heatmap: np.ndarray, row_labels: Sequence[str] | None,
        col_labels: Sequence[str] | None, path: str | Path,
    ) -> Path:
        """Heatmap of the cross-modal contribution ``[timesteps x features]``."""
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * heatmap.shape[1]), 5))
        im = ax.imshow(heatmap, cmap="magma", aspect="auto")
        fig.colorbar(im, ax=ax)
        if row_labels is not None and len(row_labels) == heatmap.shape[0]:
            ax.set_yticks(range(heatmap.shape[0]))
            ax.set_yticklabels(row_labels, fontsize=7)
        if col_labels is not None and len(col_labels) == heatmap.shape[1]:
            ax.set_xticks(range(heatmap.shape[1]))
            ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("tabular feature")
        ax.set_ylabel("observation date")
        ax.set_title("Cross-Modal Contribution")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    # ------------------------------------------------------------------ #
    # Fusion weights / gates
    # ------------------------------------------------------------------ #

    def fusion_weights(
        self, gates: Mapping[str, float], path: str | Path
    ) -> Path:
        """Horizontal bar chart of the per-modality fusion gate values.

        Args:
            gates: ``{"image_gate": ..., "tabular_gate": ...,
            "fusion_gate": ..., "temporal_gate": ...}``.
        """
        names = list(gates)
        values = [float(gates[name]) for name in names]
        fig, ax = plt.subplots(figsize=(6, max(2.0, 0.6 * len(names))))
        order = np.argsort(values)
        bars = ax.barh(
            [names[i] for i in order],
            [values[i] for i in order],
            color=["#2e7d32" if v >= 0.5 else "#c62828" for v in values],
        )
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("gate value (0 = modality suppressed)")
        ax.set_title("Fusion gate weights")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    # ------------------------------------------------------------------ #
    # Confidence / calibration
    # ------------------------------------------------------------------ #

    def confidence_distribution(
        self, confidences: np.ndarray, bins: int = 10, path: str | Path = "confidence.png"
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(np.asarray(confidences, dtype="float64"), bins=bins, range=(0, 1), color="#8c96c6")
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        ax.set_title("Confidence Distribution")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)

    def calibration_curve(
        self, bins: Mapping[str, Any], path: str | Path
    ) -> Path:
        """Reliability diagram: accuracy vs confidence per bin."""
        fig, ax = plt.subplots(figsize=(6, 5))
        confs, accs = [], []
        for center, data in sorted(bins.items(), key=lambda kv: float(kv[0])):
            confs.append(data["confidence"])
            accs.append(data["accuracy"])
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot(confs, accs, "o-", color="#e34a33", label="model")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title("Calibration (Reliability) Diagram")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return Path(path)
