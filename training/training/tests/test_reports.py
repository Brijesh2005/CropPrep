"""Report generation tests."""

from __future__ import annotations

import csv
import io

import pytest

from training.training.config import TrainingConfig
from training.training.cropfusion_trainer import CropFusionTrainingResult
from training.training.reports import (
    REPORT_TYPES,
    default_reports_dir,
    generate_reports,
    learning_curve_csv,
    training_report,
)


def _result(best_path: str | None = "ckpt/best.pt") -> CropFusionTrainingResult:
    return CropFusionTrainingResult(
        epochs=3,
        steps=6,
        history=[
            {
                "epoch": 1, "train_loss": 0.5, "val_loss": 0.6,
                "crop/accuracy": 0.7, "yield/rmse": 1.2, "lr": 1e-4,
                "stage": "tabular",
            },
            {
                "epoch": 2, "train_loss": 0.4, "val_loss": 0.5,
                "crop/accuracy": 0.8, "yield/rmse": 1.0, "lr": 5e-5,
                "stage": "tabular",
            },
            {
                "epoch": 3, "train_loss": 0.3, "val_loss": 0.4,
                "crop/accuracy": 0.85, "yield/rmse": 0.9, "lr": 2e-5,
                "stage": "finetune",
            },
        ],
        best_metrics={"val_loss": 0.4, "crop/accuracy": 0.85},
        best_path=best_path,
    )


def _config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        name="reports_test",
        general={"output_dir": str(tmp_path / "out")},
        checkpoint={"directory": str(tmp_path / "ckpt")},
    )


def test_default_reports_dir(tmp_path):
    config = TrainingConfig(general={"output_dir": str(tmp_path / "out")})
    assert default_reports_dir(config) == tmp_path / "out" / "reports"
    config = TrainingConfig(
        general={"output_dir": str(tmp_path / "out"), "reports_dir": str(tmp_path / "r")}
    )
    assert default_reports_dir(config) == tmp_path / "r"


def test_generate_reports_writes_all_types(tmp_path):
    config = _config(tmp_path)
    paths = generate_reports(config, _result())
    assert set(paths) == set(REPORT_TYPES)
    for report_type, path in paths.items():
        assert path.exists()
        assert path.read_text(encoding="utf-8") != ""


def test_training_report_content(tmp_path):
    text = training_report(_config(tmp_path), _result())
    assert "reports_test" in text
    assert "crop/accuracy" in text
    assert "ckpt/best.pt" in text


def test_validation_report_table(tmp_path):
    text = generate_reports(_config(tmp_path), _result())[
        "validation"
    ].read_text(encoding="utf-8")
    assert "val_loss" in text
    assert "crop/accuracy" in text
    assert "yield/rmse" in text


def test_metrics_report_tables(tmp_path):
    text = generate_reports(_config(tmp_path), _result())[
        "metrics"
    ].read_text(encoding="utf-8")
    assert "Classification (crop)" in text
    assert "Regression (yield)" in text
    assert "crop/accuracy" in text
    assert "yield/rmse" in text


def test_learning_curve_csv(tmp_path):
    path = generate_reports(_config(tmp_path), _result())["learning_curve"]
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    header = rows[0]
    assert header[0] == "epoch"
    assert "train_loss" in header
    assert "stage" not in header  # string column is dropped from CSV
    assert len(rows) == 4  # header + 3 epochs


def test_learning_curve_csv_drops_strings():
    text = learning_curve_csv(_result())
    assert "stage" not in text
    assert "train_loss" in text


def test_checkpoint_report_lists_artifacts(tmp_path):
    config = _config(tmp_path)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "best.pt").write_bytes(b"x")
    text = generate_reports(config, _result())[
        "checkpoint"
    ].read_text(encoding="utf-8")
    assert "best.pt" in text


def test_validation_report_no_val_data(tmp_path):
    result = CropFusionTrainingResult(epochs=1, steps=1, history=[{"epoch": 1}])
    text = generate_reports(_config(tmp_path), result)[
        "validation"
    ].read_text(encoding="utf-8")
    assert "No validation metrics" in text
