"""Automatic training visualizations.

:class:`Visualizer` renders loss / accuracy / learning-rate curves, a
regression scatter plot, a confusion matrix heatmap, precision-recall curves
and a feature-distribution plot (PCA of the shared representation), then
assembles them into a self-contained HTML training dashboard.

Uses matplotlib (Agg backend) and is fully optional — every figure is guarded
so a missing dependency only disables that chart.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # force a headless backend before pyplot is imported anywhere
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover - matplotlib optional
    pass


def _series(history: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for record in history:
        value = record.get(key)
        if value is not None and isinstance(value, (int, float)):
            values.append(float(value))
    return values


class Visualizer:
    """Render training charts + an HTML dashboard to ``output_dir``."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        history: Sequence[Mapping[str, Any]],
        evaluation: Any | None = None,
        *,
        run_name: str = "run",
    ) -> dict[str, Path]:
        """Generate every enabled chart and the dashboard.

        Args:
            history: Per-epoch metric records (from ``Trainer.history``).
            evaluation: Optional :class:`~ai.training.evaluator.EvaluationResult`.
            run_name: Display name for the dashboard.

        Returns:
            Mapping of chart name -> generated file path.
        """
        if not _matplotlib_available():
            return {"dashboard": self.build_dashboard(history, evaluation, run_name)}

        artifacts: dict[str, Path] = {}
        epochs = list(range(1, len(history) + 1))

        if _series(history, "train_loss"):
            artifacts["loss_curves"] = self.loss_curves(history, epochs)
        if _series(history, "crop/accuracy"):
            artifacts["accuracy_curves"] = self.accuracy_curves(history, epochs)
        if _series(history, "lr"):
            artifacts["lr_curves"] = self.lr_curves(history, epochs)

        if evaluation is not None:
            if evaluation.predictions.get("yield_pred") is not None:
                artifacts["regression_scatter"] = self.regression_scatter(evaluation)
            confusion = evaluation.confusion_matrix.get("crop")
            if confusion:
                artifacts["confusion_matrix"] = self.confusion_matrix(confusion)
            if evaluation.predictions.get("crop_pred") is not None:
                artifacts["precision_recall"] = self.precision_recall(evaluation)
            if evaluation.feature_embeddings is not None:
                artifacts["feature_distribution"] = self.feature_distribution(
                    evaluation.feature_embeddings, evaluation.feature_labels
                )

        artifacts["dashboard"] = self.build_dashboard(history, evaluation, run_name)
        return artifacts

    # ------------------------------------------------------------------ #
    # Individual charts
    # ------------------------------------------------------------------ #

    def loss_curves(
        self, history: Sequence[Mapping[str, Any]], epochs: Sequence[int]
    ) -> Path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        train = _series(history, "train_loss")
        val = _series(history, "val_loss")
        ax.plot(epochs[: len(train)], train, label="train")
        if val:
            ax.plot(epochs[: len(val)], val, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training / Validation Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        path = self.output_dir / "loss_curves.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def accuracy_curves(
        self, history: Sequence[Mapping[str, Any]], epochs: Sequence[int]
    ) -> Path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for key, label in (("crop/accuracy", "accuracy"), ("crop/f1", "f1")):
            values = _series(history, key)
            if values:
                ax.plot(epochs[: len(values)], values, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.set_title("Crop Classification Curves")
        ax.legend()
        ax.grid(alpha=0.3)
        path = self.output_dir / "accuracy_curves.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def lr_curves(
        self, history: Sequence[Mapping[str, Any]], epochs: Sequence[int]
    ) -> Path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        values = _series(history, "lr")
        if values:
            ax.plot(epochs[: len(values)], values)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_yscale("log")
        ax.set_title("Learning Rate Schedule")
        ax.grid(alpha=0.3)
        path = self.output_dir / "lr_curves.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def regression_scatter(self, evaluation: Any) -> Path:
        import matplotlib.pyplot as plt

        pred = np.asarray(evaluation.predictions["yield_pred"], dtype=float)
        target = np.asarray(evaluation.predictions["yield_target"], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(target, pred, alpha=0.5, s=12)
        limits = [
            min(float(target.min()), float(pred.min())),
            max(float(target.max()), float(pred.max())),
        ]
        ax.plot(limits, limits, "r--", label="perfect")
        ax.set_xlabel("actual yield")
        ax.set_ylabel("predicted yield")
        ax.set_title("Yield Regression: Predictions vs Actual")
        ax.legend()
        ax.grid(alpha=0.3)
        path = self.output_dir / "regression_scatter.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def confusion_matrix(self, matrix: Any) -> Path:
        import matplotlib.pyplot as plt

        cm = np.asarray(matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(max(6, cm.shape[0] * 0.6),
                                        max(5, cm.shape[1] * 0.5)))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title("Crop Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                        fontsize=8)
        path = self.output_dir / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def precision_recall(self, evaluation: Any) -> Path:
        import matplotlib.pyplot as plt

        pred = np.asarray(evaluation.predictions.get("crop_pred", []), dtype=int)
        target = np.asarray(evaluation.predictions.get("crop_target", []), dtype=int)
        if pred.size == 0 or target.size == 0:
            raise ValueError("empty predictions for precision-recall")
        try:
            from sklearn.metrics import precision_recall_curve
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("sklearn required for precision-recall") from exc

        num_classes = int(pred.max()) + 1 if pred.size else 1
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls in range(num_classes):
            y_true = (target == cls).astype(int)
            if y_true.sum() == 0:
                continue
            y_score = (pred == cls).astype(float)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ax.plot(recall, precision, label=f"class {cls}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title("Precision-Recall by Crop Class")
        ax.legend()
        ax.grid(alpha=0.3)
        path = self.output_dir / "precision_recall.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def feature_distribution(
        self, embeddings: np.ndarray, labels: np.ndarray | None = None
    ) -> Path:
        import matplotlib.pyplot as plt

        data = np.asarray(embeddings, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("feature embeddings must be [N, D] with D >= 2")
        try:
            from sklearn.decomposition import PCA

            reduced = PCA(n_components=2).fit_transform(data)
        except Exception:
            reduced = data[:, :2]

        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None and labels.size:
            classes = np.unique(labels)
            for cls in classes:
                mask = labels == cls
                ax.scatter(reduced[mask, 0], reduced[mask, 1], s=10,
                           label=f"class {cls}", alpha=0.6)
            ax.legend(fontsize=8)
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.6)
        ax.set_title("Shared Representation Distribution (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)
        path = self.output_dir / "feature_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # Dashboard
    # ------------------------------------------------------------------ #

    def build_dashboard(
        self,
        history: Sequence[Mapping[str, Any]],
        evaluation: Any | None = None,
        run_name: str = "run",
    ) -> Path:
        """Assemble a self-contained HTML dashboard embedding the charts."""
        images: list[tuple[str, str]] = []
        for name in (
            "loss_curves",
            "accuracy_curves",
            "lr_curves",
            "regression_scatter",
            "confusion_matrix",
            "precision_recall",
            "feature_distribution",
        ):
            path = self.output_dir / f"{name}.png"
            if path.exists():
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                images.append((name, encoded))

        rows = []
        if history:
            last = dict(history[-1])
            metrics_cells = "".join(
                f"<td>{key}</td><td>{value:.4f}</td>" for key, value in last.items()
                if isinstance(value, (int, float))
            )
            rows.append(
                f"<h3>Final epoch metrics</h3><table border='1' "
                f"style='border-collapse:collapse'>{metrics_cells}</table>"
            )
        if evaluation is not None:
            score = evaluation.multi_task_score
            rows.append(f"<p><b>Multi-task score:</b> {score:.4f}</p>")

        image_tags = "\n".join(
            f"<h3>{name.replace('_', ' ').title()}</h3>"
            f"<img src='data:image/png;base64,{encoded}' style='max-width:100%'/>"
            for name, encoded in images
        )

        html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>CropFusion — {run_name}</title></head>
<body style='font-family:system-ui,sans-serif;margin:2rem'>
<h1>CropFusion Training Dashboard — {run_name}</h1>
{''.join(rows)}
{image_tags}
</body></html>"""
        path = self.output_dir / "dashboard.html"
        path.write_text(html, encoding="utf-8")
        return path


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False
