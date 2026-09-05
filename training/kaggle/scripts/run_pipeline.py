"""Kaggle training pipeline driver — end-to-end run.

Drives the full R2.3 training chain on one machine::

    DatasetManager → ensure image (Kaggle mount / kagglehub) → generate image
    metadata → STAM.initialize() → ObservationResolver.plan()/resolve()
    → corpus → Experiment.run() (preprocess → model → trainer → evaluate)

Unlike :mod:`run_training` (readiness-only orchestration), this script actually
executes the pipeline and writes a ``pipeline.json`` report. It degrades
gracefully when the imagery dataset is not available (e.g. a research machine
without the Kaggle mount): the corpus stage still runs and training is skipped
with the reason recorded.

Run on Kaggle (imagery attached under /kaggle/input)::

    !python training/kaggle/scripts/run_pipeline.py

Run on a research machine::

    python training/kaggle/scripts/run_pipeline.py --repo-root . --skip-training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    """Force the repository root to the front of ``sys.path``.

    Called before any ``training.*`` import so ``import training`` always
    resolves to THIS repository — a stale ``/kaggle/working/training`` folder
    or a working-directory entry must not shadow the real package.
    """
    import sys

    repo_root = repo_root.resolve()
    root = str(repo_root)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    repo_training = (repo_root / "training").resolve()
    for entry in list(sys.path):
        if entry == root or entry == "":
            continue
        shadow = Path(entry) / "training"
        if shadow.exists() and shadow.resolve() != repo_training:
            print(f"[run_pipeline] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.dataset_manager import DatasetManager, load_settings  # noqa: E402
from training.kaggle.config import (  # noqa: E402
    load_kaggle_config,
    load_logging_config,
    load_paths_config,
    WorkspaceLayout,
)
from training.kaggle.frozen_corpus import (  # noqa: E402
    FrozenCorpusError,
    FrozenCorpusLoader,
)
from training.kaggle.environment import EnvironmentManager  # noqa: E402
from training.kaggle.logging import TrainingLogger  # noqa: E402
from training.kaggle.validation import TrainingValidator  # noqa: E402
from training.kaggle.workspace import WorkspaceManager  # noqa: E402
from training.models.config import (  # noqa: E402
    load_model_config as load_model_cfg,
)
from training.preprocessing import Preprocessor  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402
from training.stam.observation_resolver import (  # noqa: E402
    ObservationCorpus,
    ObservationResolver,
    ResolvedSample,
)
from training.training import Experiment  # noqa: E402
from training.training.config import (  # noqa: E402
    load_training_config as load_training_cfg,
)


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def _parse_csv_strs(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [token.strip() for token in value.split(",") if token.strip()]


def _emit_report(
    report: dict[str, Any], args: argparse.Namespace, workspace: Any
) -> Path:
    """Write the pipeline report JSON (also called on failure / skip paths).

    R5.3: the report is emitted whenever the pipeline terminates — completed,
    skipped (empty split / failed contract) or *failed* mid-training — so a
    crashed run can never leave a stale ``pipeline.json`` (or nothing) behind
    for the notebook's status cells to misread as "skipped".
    """
    output = Path(args.output) if args.output else workspace.output_path("reports")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "pipeline.json"
    target.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run_pipeline] wrote pipeline report -> {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-run-pipeline",
        description="End-to-end training pipeline driver",
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument(
        "--paths-config",
        default=str(_REPO_ROOT / "training" / "config" / "paths.yaml"),
    )
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
    )
    parser.add_argument(
        "--training-config",
        default=str(_REPO_ROOT / "training" / "config" / "training.yaml"),
    )
    parser.add_argument(
        "--model-config",
        default=str(_REPO_ROOT / "training" / "config" / "model.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"),
    )
    parser.add_argument(
        "--preprocessing-config",
        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"),
        help="Preprocessing config (image size, encoding, augmentation)",
    )
    parser.add_argument(
        "--validation-config",
        default=str(_REPO_ROOT / "training" / "config" / "validation.yaml"),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the pipeline report JSON here (default: workspace outputs)",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated years to sample (default: inferred from the catalog)",
    )
    parser.add_argument(
        "--seasons",
        default=None,
        help="Comma-separated season names to sample (default: every calendar season)",
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=None,
        help="Cap the number of distinct locations sampled",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Cap the number of (location, year, season) cells resolved (smoke runs)",
    )
    parser.add_argument(
        "--force-metadata",
        action="store_true",
        help="Regenerate imagery metadata records even when unchanged",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Resolve the corpus but do not run the Experiment",
    )
    parser.add_argument(
        "--frozen-crop-csv",
        default=None,
        help="Path to the frozen supervised crop CSV (bypasses ObservationResolver)",
    )
    parser.add_argument(
        "--frozen-manifest",
        default=None,
        help="Path to the frozen corpus manifest JSON (validates + stamps provenance)",
    )
    parser.add_argument(
        "--verify-contract",
        action="store_true",
        default=True,
        help="Verify the frozen corpus data contract before training (default: True)",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Lightweight numerical gate: one training epoch + one validation "
             "pass with the real corpus (the pre-full-run NaN/Inf check). "
             "Overrides training epochs to 1 and disables benchmark/visualization.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (e.g. 3 for a short numerical-stability "
             "test on Kaggle before committing to a full 30-epoch run).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    _add_repo_root(repo_root)

    # 1. Configuration.
    paths = load_paths_config(Path(args.paths_config))
    kaggle_cfg = load_kaggle_config()
    logging_cfg = load_logging_config()
    dataset_settings = load_settings(Path(args.dataset_config))
    training_cfg = load_training_cfg(Path(args.training_config))
    model_cfg = load_model_cfg(Path(args.model_config))
    stam_cfg = load_stam_config(Path(args.stam_config))
    preprocessing_cfg = Preprocessor.from_config(args.preprocessing_config)
    if args.lightweight:
        # R5.3 lightweight numerical gate: 1 epoch, no benchmark / visualizer.
        # Validation stays FP32 (validation.amp defaults to False) so the
        # ``B * T`` eval fast path is numerically safe; training stays AMP.
        training_cfg.train.epochs = 1
        training_cfg.benchmark.enabled = False
        training_cfg.visualization.enabled = False
        training_cfg.checkpoint.save_best = False
        training_cfg.checkpoint.save_latest = False
        print(
            "[run_pipeline] --lightweight: 1 training epoch + validation "
            "(real corpus, FP32 validation, AMP training)"
        )
    if (
        stam_cfg.temporal.season_file is not None
        and not stam_cfg.temporal.season_file.is_absolute()
    ):
        # Repo-relative calendar reference: resolve against the STAM config's
        # own directory (training/config/) so it works regardless of CWD.
        stam_cfg.temporal.season_file = (
            Path(args.stam_config).resolve().parent / stam_cfg.temporal.season_file
        )

    if args.epochs is not None:
        training_cfg.train.epochs = args.epochs

    # 2. Environment + logging + workspace.
    environment = EnvironmentManager(repo_root)
    env_report = environment.report()
    layout = WorkspaceLayout.resolve(paths, repo_root=repo_root)
    logger = TrainingLogger(logging_cfg, log_dir=layout.logs).setup()
    workspace = WorkspaceManager(layout)
    workspace.create()
    logger.log_experiment(
        "pipeline_start",
        repo_root=str(repo_root),
        gpu=env_report["gpu"].get("available"),
    )

    report: dict[str, Any] = {
        "environment": env_report,
        "configuration": {
            "paths": paths.model_dump(),
            "kaggle": kaggle_cfg.model_dump(),
            "dataset_config": args.dataset_config,
            "training_config": args.training_config,
            "model_config": args.model_config,
            "stam_config": args.stam_config,
            "preprocessing_config": args.preprocessing_config,
        },
        "workspace": workspace.report(),
    }

    # 3. Dataset Manager.
    manager = DatasetManager(dataset_settings)
    try:
        manifests = manager.provider_manifests()
        report["dataset_manager"] = {
            "providers": manifests,
            "tabular_datasets": manager.tabular_names(),
        }

        # 4. Imagery (Kaggle mount preferred; kagglehub fallback).
        image_manifest = manifests.get("kaggle_hub_image", {})
        imagery_available = image_manifest.get("available") is True
        image_stage: dict[str, Any] = {"available": imagery_available}
        if imagery_available:
            root = manager.ensure_image()
            image_stage["source"] = (
                "mount" if str(root).startswith("/kaggle/input") else "catalog"
            )
            image_stage["root"] = str(root)
            count = manager.generate_image_metadata(force=args.force_metadata)
            image_stage["metadata_records"] = count
        else:
            image_stage["warning"] = (
                "imagery unavailable (Kaggle mount or materialised catalog required)"
            )
        report["imagery"] = image_stage

        # 5. STAM.
        stam = STAM(manager, stam_cfg)
        stam.initialize()
        report["stam"] = {
            "initialized": True,
            "matcher": stam.matcher.spatial_stats(),
            "seasons": stam.season_resolver.names(),
            "season_calendar": stam.season_resolver.source,
        }

        # 6. Corpus — frozen CSV or ObservationResolver.
        frozen_csv = args.frozen_crop_csv
        frozen_manifest = args.frozen_manifest
        use_frozen = frozen_csv is not None

        if use_frozen:
            # Frozen corpus path (R5.2.7 supervised crop training).
            frozen_loader = FrozenCorpusLoader(
                csv_path=frozen_csv,
                manifest_path=frozen_manifest or str(
                    _REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"
                ),
            )
            frozen_loader.validate()

            # R5.4 explicit class contract: drive the label vocabulary from the
            # manifest instead of letting the encoder emerge from whichever
            # labels appear in the (quality-filtered) training split. This is
            # the guard that stops a rare label silently resizing the crop head.
            declared = list(frozen_loader.manifest.get("supervised_classes") or [])
            excluded = list(frozen_loader.manifest.get("excluded_classes") or [])
            preprocessing_cfg.config.label.declared_classes = declared
            preprocessing_cfg.config.label.excluded_classes = excluded
            print("\n--- Class schema (explicit contract) ---")
            print(f"  supervised_classes ({len(declared)}): {declared}")
            print(f"  excluded_classes: {excluded or []}")
            manifest_train_counts = frozen_loader.manifest.get(
                "class_counts", {}
            ).get("train", {})
            schema_warnings = [
                f"class '{c}' declared supervised but has ZERO train samples "
                "(classifier cannot learn it)"
                for c in declared
                if not (manifest_train_counts.get(c, 0) > 0)
            ]
            for warn in schema_warnings:
                print(f"  [WARN] {warn}")

            train_obs, val_obs, test_obs = frozen_loader.build(stam)
            accepted = train_obs + val_obs + test_obs

            # Post-build class -> crop-head index -> per-split counts printout.
            print("\n--- Class -> crop-head index -> split counts ---")
            for idx, cls in enumerate(declared):
                counts = {
                    part: sum(1 for o in obs if o.crop == cls)
                    for part, obs in (
                        ("train", train_obs),
                        ("val", val_obs),
                        ("test", test_obs),
                    )
                }
                print(
                    f"  [{idx}] {cls:10s} train={counts['train']:5d} "
                    f"val={counts['val']:4d} test={counts['test']:4d}"
                )

            # R5.2 guard: every split must be non-empty before Phase 4 runs.
            if not (train_obs and val_obs and test_obs):
                report["training"] = {
                    "status": "skipped",
                    "reason": "frozen corpus produced an empty split",
                    "train": len(train_obs),
                    "val": len(val_obs),
                    "test": len(test_obs),
                    "build_stats": getattr(frozen_loader, "last_build_stats", None),
                }
                print("\n[FATAL] Frozen corpus produced an empty split:")
                print(
                    f"  train={len(train_obs)} val={len(val_obs)} test={len(test_obs)}"
                )
                print("Training aborted.\n")
                _emit_report(report, args, workspace)
                return 1

            corpus_path = workspace.output_path("reports", "frozen_corpus.json")

            # Persist the frozen corpus in ObservationCorpus JSON format so
            # the post-training verification scripts (verify_multimodal_tensors,
            # verify_split_composition) can load it via --corpus.
            ObservationCorpus(
                samples=[
                    ResolvedSample(
                        location_id=o.provenance.get("record_id")
                        or str(o.observation_id),
                        name=o.crop or "unknown",
                        lon=o.location.lon,
                        lat=o.location.lat,
                        year=o.temporal.year or 0,
                        season=o.temporal.season or "unknown",
                        status="accepted",
                        quality_score=o.quality.overall_score,
                        observation=o,
                    )
                    for o in accepted
                ],
                config={"source": "frozen_crop_supervised_v1"},
            ).save(corpus_path)

            # Data contract printout + verification (Kaggle stop-on-mismatch).
            contract = frozen_loader.data_contract_printout(
                train_obs, val_obs, test_obs
            )
            contract_passed, contract_errors = frozen_loader.verify_contract(
                contract, train_obs, val_obs, test_obs
            )
            if not contract_passed:
                report["training"] = {
                    "status": "skipped",
                    "reason": "frozen corpus data contract FAILED verification",
                    "contract_errors": contract_errors,
                    "accepted_observations": len(accepted),
                }
                print("\n[FATAL] Frozen corpus data contract verification FAILED:")
                for err in contract_errors:
                    print(f"  - {err}")
                print("Training aborted.\n")
                _emit_report(report, args, workspace)
                return 1

            report["corpus"] = {
                "type": "frozen_crop_supervised_v1",
                "version": contract.get("version"),
                "manifest_checksum": contract.get("manifest_checksum"),
                "total": len(accepted),
                "train": len(train_obs),
                "val": len(val_obs),
                "test": len(test_obs),
                "class_counts": contract.get("overall_class_counts"),
                "split_strategy": contract.get("split_strategy"),
                "split_groups": contract.get("split_groups"),
                "class_schema": {
                    "supervised_classes": declared,
                    "excluded_classes": excluded,
                },
                "path": str(corpus_path),
            }
            report["corpus"]["build_stats"] = getattr(
                frozen_loader, "last_build_stats", None
            )
            report["corpus"]["imagery"] = frozen_loader.imagery_summary(
                train_obs, val_obs, test_obs,
                max_observations=preprocessing_cfg.config.temporal.max_observations,
            )
            report["corpus"]["imagery"]["imagery_window"] = {
                "mode": stam_cfg.imagery.mode,
                "window_days": stam_cfg.imagery.window_days,
                "start_month": stam_cfg.imagery.start_month,
                "span_months": stam_cfg.imagery.span_months,
                "max_observations": stam_cfg.imagery.max_observations,
                "strategy": stam_cfg.imagery.strategy,
                "temporal_pad_slots": preprocessing_cfg.config.temporal.max_observations,
                "description": (
                    "season-calendar window (legacy)"
                    if stam_cfg.imagery.mode == "season"
                    else f"{stam_cfg.imagery.mode} window"
                ),
            }
            # Corpus-level real-vs-zero-filled slot statistics (R5.3): proves
            # every accepted split carries real NDVI/EVI frames and quantifies
            # the zero-fill padding within the fixed temporal window.
            imagery_frames = frozen_loader.corpus_imagery_diagnostics(
                train_obs, val_obs, test_obs,
                max_observations=preprocessing_cfg.config.temporal.max_observations,
            )
            report["corpus"]["imagery"]["frames"] = imagery_frames
            print("\n--- Corpus Imagery (real vs zero-filled slots) ---")
            for part in ("train", "val", "test", "overall"):
                block = imagery_frames[part]
                print(f"  {part:8s} samples={block['samples']}")
                for stream in ("ndvi", "evi"):
                    s = block["streams"][stream]
                    print(
                        f"    {stream:5s} slots={s['total_slots']} "
                        f"real={s['real_slots']} zero_fill={s['zero_filled_slots']} "
                        f"real_frac={s['real_frac']:.1%} "
                        f"samples_no_imagery={s['samples_without_imagery']}"
                    )
                fs = block.get("real_frames_per_sample", {})
                print(
                    f"    real frames/sample: mean={fs.get('mean')} "
                    f"median={fs.get('median')} p25={fs.get('p25')} "
                    f"p75={fs.get('p75')} max={fs.get('max')} "
                    f"dist={block.get('real_frames_distribution', {})}"
                )
            # R5.3 (supersedes the earlier single-frame note): the number of
            # real temporal frames per observation is determined by the
            # configured imagery acquisition window (ST_IMAGERY__*), NOT by
            # construction. The season calendar window (legacy) resolves one
            # composite per Kharif survey on the Kaggle seasonal-composite
            # mount; a window_days / crop_year window recovers multiple real
            # frames. Slots left empty after trimming real frames are
            # zero-filled padding, which the temporal mask excludes.
            print(
                f"  Imagery window: mode={stam_cfg.imagery.mode} "
                f"strategy={stam_cfg.imagery.strategy} "
                f"imagery_max_frames={stam_cfg.imagery.max_observations} "
                f"temporal_pad_slots={preprocessing_cfg.config.temporal.max_observations}"
            )
            logger.log_experiment(
                "frozen_corpus_loaded",
                total=len(accepted),
                train=len(train_obs),
                val=len(val_obs),
                test=len(test_obs),
            )
        else:
            # Standard path: ObservationResolver builds the corpus.
            resolver = ObservationResolver(stam)
            plan = resolver.plan(
                years=_parse_csv_ints(args.years),
                seasons=_parse_csv_strs(args.seasons),
                max_locations=args.max_locations,
            )
            if args.max_cells and plan.total > args.max_cells:
                plan = plan.model_copy(update={"cells": plan.cells[: args.max_cells]})
            corpus = resolver.resolve(plan)
            corpus_path = workspace.output_path("reports", "corpus.json")
            corpus.save(corpus_path)
            report["corpus"] = {
                **corpus.summary(),
                "path": str(corpus_path),
                "plan": plan.counts(),
            }
            logger.log_experiment(
                "corpus_resolved",
                total=corpus.total,
                **corpus.status_counts(),
            )
            accepted = corpus.accepted_observations()
            train_obs = val_obs = test_obs = None

        # 7. Experiment (skipped without accepted observations or by flag).
        if args.skip_training or not accepted:
            report["training"] = {
                "status": "skipped",
                "reason": (
                    "skip_training flag" if args.skip_training
                    else "no accepted observations (imagery/locations unavailable)"
                ),
                "accepted_observations": len(accepted),
            }
        else:
            run_dir = workspace.run_output(training_cfg.name)
            pre_split = (
                (train_obs, val_obs, test_obs)
                if use_frozen and train_obs is not None
                else None
            )
            experiment = Experiment(
                training_cfg,
                accepted,
                extractor=stam.get_patch,
                model_config=model_cfg,
                preprocessor=preprocessing_cfg,
                run_dir=run_dir,
                run_name=training_cfg.name,
                pre_split=pre_split,
            )
            # R5.3: a mid-training crash (e.g. ValidationError from a
            # non-finite validation loss) must record a truthful report instead
            # of leaving the workspace without pipeline.json.
            try:
                result = experiment.run()
            except Exception as exc:
                report["training"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "accepted_observations": len(accepted),
                    "run_dir": str(run_dir),
                }
                _emit_report(report, args, workspace)
                logger.log_experiment(
                    "training_failed", run_dir=str(run_dir), error=str(exc)
                )
                raise
            report["training"] = {
                "status": "completed",
                "accepted_observations": len(accepted),
                "run_dir": str(run_dir),
                "report": result.to_dict(),
            }
            logger.log_experiment(
                "training_complete",
                run_dir=str(run_dir),
                metrics=result.metrics,
            )
    finally:
        manager.close()

    # 8. Validation (config / python / gpu / deps / folders / disk).
    validator = TrainingValidator(paths, layout, env_report)
    validation = validator.validate(provider_manifests=manifests)
    report["validation"] = validation.to_dict()
    logger.log_experiment(
        "pipeline_complete",
        passed=validation.passed,
        severity_summary=validation.by_severity(),
    )

    _emit_report(report, args, workspace)
    if use_frozen:
        print(
            f"[run_pipeline] frozen corpus -> total={len(accepted)} "
            f"train={len(train_obs)} val={len(val_obs)} test={len(test_obs)}"
        )
    else:
        print(
            f"[run_pipeline] corpus summary -> total={corpus.total} "
            f"accepted={len(corpus.accepted())} rejected={len(corpus.rejected())} "
            f"errors={len(corpus.errors())}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
