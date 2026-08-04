"""Experiment orchestration.

:class:`Experiment` wires the whole pipeline for one run:

    observations → split/CV → fit preprocessor (train only) → loaders →
    build model (from preprocessor) → trainer → evaluate on test → benchmark
    → visualize → write artifacts.

Nothing is read directly from disk: the Phase 4 preprocessing layer (Dataset
Manager → STAM → preprocessing) supplies the observations and a patch
extractor.

:class:`ExperimentReport` carries the metrics, artifacts and configuration of
one run so callers can compare runs (e.g. ablations).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ai.models import ModelConfig, ModelFactory
from ai.preprocessing import (
    CropFusionDataset,
    DataloaderConfig,
    Preprocessor,
    build_dataloader,
    split_observations,
)
from services.dataset_manager.config import deep_merge

from .benchmark import Benchmark
from .callbacks import HistoryRecorder
from .checkpoint import TrainingCheckpointManager
from .config import TrainingConfig
from .evaluator import Evaluator, EvaluationResult
from .exceptions import ValidationError
from .logger import ExperimentLogger
from .trainer import Trainer, TrainingResult
from .validator import cross_validation_splits
from .visualizer import Visualizer


@dataclass
class ExperimentReport:
    """Outcome of one :class:`Experiment` run."""

    run_name: str
    run_dir: Path
    training: TrainingResult | None = None
    evaluation: EvaluationResult | None = None
    benchmark: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, Any]:
        if self.evaluation is not None:
            return self.evaluation.to_dict()
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "training": self.training.summary() if self.training else {},
            "evaluation": self.evaluation.to_dict() if self.evaluation else {},
            "benchmark": self.benchmark,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
        }


class Experiment:
    """Run a full training + evaluation experiment."""

    def __init__(
        self,
        training_config: TrainingConfig,
        observations: Sequence[Any],
        *,
        preprocessor: Preprocessor | None = None,
        extractor: Any | None = None,
        model_config: ModelConfig | Mapping[str, Any] | None = None,
        model_config_overrides: Mapping[str, Any] | None = None,
        run_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> None:
        self.config = training_config
        self.observations = list(observations)
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.model_config = model_config
        self.model_config_overrides = dict(model_config_overrides or {})
        self.run_name = run_name or training_config.name
        self.run_dir = Path(run_dir or (training_config.general.output_dir / self.run_name))

        self.device = torch.device("cpu")
        if training_config.general.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif training_config.general.device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(training_config.general.device)
        else:
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def run(self) -> ExperimentReport:
        """Execute the experiment (hold-out or cross-validation)."""
        strategy = self.config.validation.strategy
        if strategy == "holdout":
            report = self._run_holdout()
        else:
            report = self._run_cross_validation()
        return report

    # ------------------------------------------------------------------ #
    # Hold-out
    # ------------------------------------------------------------------ #

    def _run_holdout(self) -> ExperimentReport:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger(self.run_dir, name=self.run_name)

        train, val, test = self._holdout_split()
        if not train or not test:
            raise ValidationError("hold-out split produced empty train/test sets")

        preprocessor = self._ensure_fitted(train)
        model_config = self._resolve_model_config(preprocessor)

        train_loader = self._build_loader(preprocessor, train, split="train")
        val_loader = self._build_loader(preprocessor, val, split="val") if val else None
        test_loader = self._build_loader(preprocessor, test, split="test")

        model = ModelFactory.create(model_config)
        training = self._run_trainer(
            model, train_loader, val_loader, logger, model_config, preprocessor,
            run_dir=self.run_dir,
        )

        evaluation = self._evaluate(model, test_loader, logger)
        benchmark = self._benchmark(model, logger)

        artifacts = self._visualize(training, evaluation, logger)
        logger.save_config_snapshot(
            self.config, model_config, preprocessor.config if preprocessor else None
        )
        logger.save_environment()
        if self.config.logging.git_hash:
            logger.save_git()

        report = ExperimentReport(
            run_name=self.run_name,
            run_dir=self.run_dir,
            training=training,
            evaluation=evaluation,
            benchmark=benchmark,
            artifacts=artifacts,
        )
        report.config_snapshot = {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "training_config": self.config.model_dump(),
            "artifacts": {k: str(v) for k, v in artifacts.items()},
        }
        return report

    # ------------------------------------------------------------------ #
    # Cross-validation
    # ------------------------------------------------------------------ #

    def _run_cross_validation(self) -> ExperimentReport:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger(self.run_dir, name=self.run_name)

        splits = cross_validation_splits(self.observations, self.config.validation)
        if not splits:
            raise ValidationError("cross-validation produced no folds")

        fold_metrics: list[dict[str, Any]] = []
        best_fold: dict[str, Any] | None = None
        for fold_index, (train, val) in enumerate(splits):
            if not train or not val:
                logger.warning("skipping empty fold", fold=fold_index)
                continue
            fold_dir = self.run_dir / f"fold_{fold_index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_logger = ExperimentLogger(fold_dir, name=f"{self.run_name}/fold_{fold_index}")

            preprocessor = self._fit_fresh(train)
            num_classes = int(getattr(preprocessor.label, "num_classes", 0) or 0)
            if num_classes < 2:
                logger.warning(
                    "skipping fold with fewer than 2 crop classes", fold=fold_index
                )
                continue
            model_config = self._resolve_model_config(preprocessor)
            train_loader = self._build_loader(preprocessor, train, split="train")
            val_loader = self._build_loader(preprocessor, val, split="val")

            model = ModelFactory.create(model_config)
            training = self._run_trainer(
                model, train_loader, val_loader, fold_logger, model_config, preprocessor,
                run_dir=fold_dir,
            )
            evaluation = self._evaluate(model, val_loader, fold_logger)

            metrics = {
                "fold": fold_index,
                "train_samples": len(train),
                "val_samples": len(val),
            }
            for key, value in evaluation.metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
            metrics["multi_task_score"] = evaluation.multi_task_score
            fold_metrics.append(metrics)
            if best_fold is None or self._is_better(metrics, best_fold):
                best_fold = metrics

            shutil.rmtree(fold_dir, ignore_errors=True)

        # Aggregated report over folds.
        agg = self._aggregate_folds(fold_metrics)
        report = ExperimentReport(
            run_name=self.run_name,
            run_dir=self.run_dir,
            benchmark={},
            artifacts={},
            evaluation=EvaluationResult(
                metrics={k: v for k, v in agg.items() if k != "fold"},
                multi_task_score=float(agg.get("multi_task_score", 0.0)),
            ),
        )
        report.config_snapshot = {
            "validation_strategy": self.config.validation.strategy,
            "k_folds": self.config.validation.k_folds,
            "fold_metrics": fold_metrics,
            "aggregate": agg,
        }
        import json

        (self.run_dir / "cross_validation.json").write_text(
            json.dumps(report.config_snapshot, indent=2), encoding="utf-8"
        )
        return report

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _holdout_split(self) -> tuple[list[Any], list[Any], list[Any]]:
        if self.config.validation.strategy != "holdout":
            raise ValidationError("_holdout_split called for a non-holdout strategy")
        # Split config lives on the preprocessor (Phase 4); default temporal.
        split_config = None
        if self.preprocessor is not None:
            split_config = self.preprocessor.config.split
        return split_observations(self.observations, split_config)

    def _ensure_fitted(self, train: Sequence[Any]) -> Preprocessor:
        if self.preprocessor is not None and getattr(self.preprocessor, "fitted", False):
            return self.preprocessor
        return self._fit_fresh(train)

    def _fit_fresh(self, train: Sequence[Any]) -> Preprocessor:
        config = self.preprocessor.config if self.preprocessor is not None else None
        preprocessor = Preprocessor(config)
        accepted, _ = preprocessor.filter(train)
        if not accepted:
            raise ValidationError("no observations survived quality filtering")
        preprocessor.fit(accepted, extractor=self.extractor)
        return preprocessor

    def _resolve_model_config(self, preprocessor: Preprocessor) -> ModelConfig:
        if self.model_config is None:
            return ModelConfig.from_preprocessor(
                preprocessor, **self.model_config_overrides
            )
        # Preserve the user's architecture settings but re-derive the
        # schema-owned fields (tabular dims, class count, image size, temporal
        # capacity) from THIS preprocessor — required for cross-validation
        # where every fold fits its own schema.
        merged = deep_merge(self._architecture_only(), self.model_config_overrides)
        return ModelConfig.from_preprocessor(preprocessor, **merged)

    def _architecture_only(self) -> dict[str, Any]:
        """The user model config with schema-owned fields removed.

        Removing (rather than nulling) lets ``ModelConfig.from_preprocessor``
        re-derive tabular dims, class count, image size and temporal capacity
        from the given preprocessor (required for cross-validation where every
        fold fits its own schema).
        """
        user = (
            self.model_config.model_dump()
            if isinstance(self.model_config, ModelConfig)
            else dict(self.model_config)
        )
        tabular = dict(user.get("tabular") or {})
        tabular.pop("numeric_dim", None)
        tabular.pop("categorical_cardinalities", None)
        user["tabular"] = tabular

        heads = dict(user.get("heads") or {})
        crop = dict(heads.get("crop") or {})
        crop.pop("num_classes", None)
        heads["crop"] = crop
        user["heads"] = heads

        image_encoder = dict(user.get("image_encoder") or {})
        image_encoder.pop("input_size", None)
        user["image_encoder"] = image_encoder

        temporal = dict(user.get("temporal") or {})
        temporal.pop("max_len", None)
        user["temporal"] = temporal
        return user

    def _build_loader(
        self, preprocessor: Preprocessor, observations: Sequence[Any], *, split: str
    ) -> Any:
        data = self.config.data
        loader_config = DataloaderConfig(
            batch_size=data.batch_size,
            workers=data.workers,
            pin_memory=data.pin_memory,
            prefetch_factor=data.prefetch_factor,
            persistent_workers=data.persistent_workers,
            shuffle_train=data.train_shuffle,
        )
        dataset = CropFusionDataset.build(
            preprocessor, observations, split=split, extractor=self.extractor
        )
        return build_dataloader(dataset, loader_config, split=split)

    def _run_trainer(
        self,
        model: torch.nn.Module,
        train_loader: Any,
        val_loader: Any | None,
        logger: ExperimentLogger,
        model_config: ModelConfig,
        preprocessor: Preprocessor,
        *,
        run_dir: str | Path,
    ) -> TrainingResult:
        from .checkpoint import TrainingCheckpointManager

        history = HistoryRecorder()
        # Scope checkpoints under the run/fold directory so every experiment's
        # artifacts are self-contained.
        checkpoint_manager = TrainingCheckpointManager(
            Path(run_dir) / "checkpoints",
            keep_last=self.config.checkpoint.keep_last,
        )
        trainer = Trainer(
            model,
            train_loader,
            self.config,
            val_loader=val_loader,
            callbacks=[history],
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            device=self.device,
        )
        result = trainer.train()
        # Move the per-run history into the report for visualization.
        result.history = history.history or result.history
        return result

    def _evaluate(
        self, model: torch.nn.Module, test_loader: Any, logger: ExperimentLogger
    ) -> EvaluationResult:
        evaluator = Evaluator(model, device=self.device, metrics_config=self.config.metrics)
        evaluation = evaluator.evaluate(test_loader)
        return evaluation

    def _benchmark(self, model: torch.nn.Module, logger: ExperimentLogger) -> dict[str, Any]:
        if not self.config.benchmark.enabled:
            return {}
        benchmark = Benchmark(
            model,
            device=self.device,
            batch_size=self.config.benchmark.batch_size,
            iterations=self.config.benchmark.iterations,
            warmup_iterations=self.config.benchmark.warmup_iterations,
            metrics_config=self.config.metrics,
        )
        sample_batch = model.sample_batch(batch_size=self.config.benchmark.batch_size)
        report = benchmark.run(
            measure_training=self.config.benchmark.measure_training_speed,
            measure_inference=self.config.benchmark.measure_inference_speed,
            sample_batch=sample_batch,
        )
        return report.to_dict()

    def _visualize(
        self,
        training: TrainingResult | None,
        evaluation: EvaluationResult | None,
        logger: ExperimentLogger,
    ) -> dict[str, Path]:
        if not self.config.visualization.enabled:
            return {}
        visualizer = Visualizer(self.config.visualization.directory)
        history = training.history if training else []
        return visualizer.visualize(history, evaluation, run_name=self.run_name)

    # ------------------------------------------------------------------ #
    # CV aggregation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        metric = candidate.get("multi_task_score", 0.0)
        return metric > current.get("multi_task_score", 0.0)

    @staticmethod
    def _aggregate_folds(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
        if not fold_metrics:
            return {}
        keys = {key for fold in fold_metrics for key in fold if isinstance(fold[key], (int, float))}
        aggregate: dict[str, float] = {}
        for key in sorted(keys):
            values = [fold[key] for fold in fold_metrics if key in fold]
            if values:
                aggregate[key] = float(sum(values) / len(values))
        return aggregate


def run_experiment(
    training_config: TrainingConfig,
    observations: Sequence[Any],
    **kwargs: Any,
) -> ExperimentReport:
    """Convenience wrapper: build an :class:`Experiment` and run it."""
    return Experiment(training_config, observations, **kwargs).run()
