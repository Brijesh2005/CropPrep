"""Visualizer tests: charts and dashboard are written to disk."""

from __future__ import annotations

import numpy as np

from training.training import Evaluator, Visualizer


def _history():
    return [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "lr": 1e-3,
         "crop/accuracy": 0.5},
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.9, "lr": 8e-4,
         "crop/accuracy": 0.6},
    ]


def test_visualize_charts(tmp_path):
    visualizer = Visualizer(tmp_path)
    artifacts = visualizer.visualize(_history())
    assert "loss_curves" in artifacts
    assert "dashboard" in artifacts
    assert artifacts["loss_curves"].exists()
    assert artifacts["dashboard"].exists()
    assert "dashboard.html" in artifacts["dashboard"].name


def test_visualize_with_evaluation(tmp_path, tabular_model, train_config):
    import torch

    from training.training import MultiTaskLoss

    loader = _eval_loader()
    evaluator = Evaluator(tabular_model)
    evaluation = evaluator.evaluate(loader, MultiTaskLoss(train_config.loss))
    visualizer = Visualizer(tmp_path)
    artifacts = visualizer.visualize(_history(), evaluation)
    for key in ("regression_scatter", "feature_distribution", "dashboard"):
        assert key in artifacts, f"missing artifact {key}"


def _eval_loader():
    import torch

    class _Fake:
        def __len__(self) -> int:
            return 2

        def __iter__(self):
            return iter(
                [
                    {"tabular": torch.randn(4, 5),
                     "crop_label": torch.randint(0, 3, (4,)),
                     "yield_label": torch.randn(4, 1)}
                    for _ in range(2)
                ]
            )

    return _Fake()
