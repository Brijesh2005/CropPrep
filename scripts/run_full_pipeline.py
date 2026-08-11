#!/usr/bin/env python3
"""Orchestrate the CropPrep Kaggle pipeline end-to-end.

Runs the five stages that take a bare repo to a downloadable release:

    1. system_check - push + run ``cropfusion-system-check`` and require the
       validation report to pass with no error-severity issues.
    2. train        - push + run ``cropfusion-train`` and require the pipeline
       report to show ``training.status == "completed"`` with at least one
       accepted corpus observation; then download the checkpoint the kernel
       registered in ``checkpoint.json``.
    3. handoff      - upload that checkpoint as a private Kaggle dataset
       (``<owner>/cropfusion-checkpoints``) and wait until it is ``complete``.
    4. export       - push + run ``cropfusion-export`` (which reads the
       checkpoint dataset from ``/kaggle/input``) and require all four release
       artifacts (``release.json``, ``model.torchscript.pt``, ``model.onnx``,
       ``model.yaml``) plus the assembled ``cropfusion_release-v<version>/``
       package (12 files) in its kernel output.
    5. summary      - print and persist a combined pipeline summary.

Any stage failure aborts immediately: the specific reason is pulled from that
stage's own report (e.g. the validation issues list, the ``training.status`` /
``reason`` field, the corpus acceptance breakdown), printed, and the script
exits non-zero. The full kernel log of every stage is saved under
``kaggle_runs/<run_id>/``.

Usage::

    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --train-timeout 10800 --runs-dir kaggle_runs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(  # type: ignore[union-attr]
        encoding="utf-8", errors="replace"
    )
    sys.stderr.reconfigure(  # type: ignore[union-attr]
        encoding="utf-8", errors="replace"
    )
except Exception:  # noqa: S110 - non-interactive streams may not support reconfigure
    pass

import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_kaggle_notebook import (  # noqa: E402 - sibling import after path fix
    NOTEBOOKS,
    REPORT_PATHS,
    build_push_dir,
    download_paths,
    parse_log,
    push_notebook,
    wait_for_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Kernel-output-relative path of the checkpoint report written by train.ipynb.
CHECKPOINT_REL = "CropPrep/training/kaggle/outputs/reports/checkpoint.json"

#: Name -> kernel-output-relative paths of the four release artifacts.
RELEASE_PATHS = {
    "release.json": "CropPrep/training/artifacts/releases/release.json",
    "model.torchscript.pt": (
        "CropPrep/training/artifacts/releases/model.torchscript.pt"
    ),
    "model.onnx": "CropPrep/training/artifacts/releases/model.onnx",
    "model.yaml": "CropPrep/training/artifacts/releases/model.yaml",
}

#: Private dataset that hands the trained checkpoint to the export kernel.
CHECKPOINT_DATASET_SLUG = "cropfusion-checkpoints"

#: Kernel-output-relative subdir where train.ipynb writes release_sources/.
RELEASE_SOURCES_REL = "CropPrep/training/artifacts/release_sources"

#: Name -> kernel-output-relative path of every train-side release source.
RELEASE_SOURCES_PATHS = {
    "preprocess/scaler.pkl": f"{RELEASE_SOURCES_REL}/preprocess/scaler.pkl",
    "preprocess/label_encoder.pkl": f"{RELEASE_SOURCES_REL}/preprocess/label_encoder.pkl",
    "metadata/metadata.db": f"{RELEASE_SOURCES_REL}/metadata/metadata.db",
    "metadata/historical_context.parquet": (
        f"{RELEASE_SOURCES_REL}/metadata/historical_context.parquet"
    ),
    "metadata/location_index.parquet": (
        f"{RELEASE_SOURCES_REL}/metadata/location_index.parquet"
    ),
    "metadata/village_metadata.parquet": (
        f"{RELEASE_SOURCES_REL}/metadata/village_metadata.parquet"
    ),
    "reports/metrics.json": f"{RELEASE_SOURCES_REL}/reports/metrics.json",
    "sources.json": f"{RELEASE_SOURCES_REL}/sources.json",
}

#: Relative files of the Prediction Platform release package (mirrors the
#: app's ``RELEASE_PACKAGE_FILES`` in application/inference_package/release/).
RELEASE_PACKAGE_FILES = (
    "model/cropfusion.pt",
    "metadata/metadata.db",
    "metadata/historical_context.parquet",
    "metadata/location_index.parquet",
    "metadata/village_metadata.parquet",
    "preprocess/scaler.pkl",
    "preprocess/label_encoder.pkl",
    "configs/model.yaml",
    "configs/inference.yaml",
    "version/manifest.json",
    "version/checksum.json",
    "reports/metrics.json",
)

#: Kernel-output-relative root of the release package directory.
RELEASE_PACKAGE_REL = "CropPrep/training/artifacts/releases"


def release_package_paths(version: str) -> dict[str, str]:
    """Name -> kernel-output-relative paths of one release package."""
    root = f"{RELEASE_PACKAGE_REL}/cropfusion_release-v{version}"
    return {
        f"release/{rel}": f"{root}/{rel}" for rel in RELEASE_PACKAGE_FILES
    }

#: Default max wall-clock wait (s) per kernel stage.
STAGE_TIMEOUTS = {"system_check": 2700.0, "train": 7200.0, "export": 3600.0}


class StageFailure(RuntimeError):
    """A pipeline stage failed; ``str(exc)`` is the report-derived reason."""


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def _load_json(path: Path | None, label: str) -> dict:
    if not path:
        raise StageFailure(f"{label} was not produced by the kernel")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report corruption -> actionable error
        raise StageFailure(f"{label} is unreadable/corrupt: {exc}") from exc


def _describe_issues(issues: list) -> str:
    if not issues:
        return "no issues listed"
    return "; ".join(
        f"{i.get('code', '?')} - {i.get('message', '(no message)')}" for i in issues
    )


def _run_stage(
    api: KaggleApi,
    owner: str,
    name: str,
    runs_dir: Path,
    timeouts: dict[str, float],
    poll_interval: float,
    keep_push_dir: bool,
    *,
    extra_datasets: tuple[str, ...] = (),
    download_map: dict[str, str] | None = None,
) -> dict:
    """Push + run one kernel, download its output files, save its log and a
    per-stage summary, and return a stage dict for the orchestrator."""
    notebook = NOTEBOOKS[name]
    kernel_ref = f"{owner}/{notebook['slug']}"
    timeout = timeouts[name]
    print(f"\n===== STAGE {name} =====  kernel {kernel_ref}")

    try:
        push_dir = build_push_dir(
            notebook, owner, extra_dataset_sources=extra_datasets,
            keep=keep_push_dir,
        )
        version, url = push_notebook(api, push_dir)
        slug_base = notebook["slug"].replace("cropfusion-", "")
        run_id = (
            f"{slug_base}-v{version}"
            if version
            else f"{slug_base}-{_utc_stamp()}"
        )
        print(f"[push] {kernel_ref} v{version or 'new'} -> {url}")
        status = wait_for_completion(api, kernel_ref, timeout, poll_interval)
    except Exception as exc:  # noqa: BLE001 - any push/run error kills the stage
        raise StageFailure(f"stage '{name}' infrastructure failure: {exc}") from exc

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = download_paths(kernel_ref, dict(download_map or REPORT_PATHS), run_dir)

    log_text = api.kernels_logs(kernel_ref)
    log_info = parse_log(log_text)
    (run_dir / "kernel.log").write_text(log_text, encoding="utf-8")

    stage = {
        "name": name,
        "run_id": run_id,
        "run_dir": run_dir,
        "kernel_ref": kernel_ref,
        "version": version,
        "url": url,
        "status": status,
        "reports": reports,
        "log": log_info,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "stage": name,
                "run_id": run_id,
                "kernel": kernel_ref,
                "kernel_version": version,
                "kernel_url": url,
                "kernel_status": status,
                "log": log_info,
                "reports": {
                    k: (str(v) if v else None) for k, v in reports.items()
                },
                "finished_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stage


def _run_export_with_retry(
    api: KaggleApi,
    owner: str,
    runs_dir: Path,
    timeouts: dict[str, float],
    poll_interval: float,
    keep_push_dir: bool,
    checkpoint_ref: str,
    download_map: dict[str, str],
    *,
    max_attempts: int = 3,
    retry_delay: float = 300.0,
) -> dict:
    """Push + run the export kernel, retrying on infrastructure failure.

    Kaggle's input mirror can lag the dataset-status/file-listing APIs, so a
    freshly uploaded checkpoint is occasionally missing from the export
    kernel's ``/kaggle/input`` mount. The kernel then dies with status ERROR
    before producing any release artifact. Re-pushing after a cooldown lets the
    mirror sync; export is idempotent, so a retry is safe. Only
    push/run/timeout failures are retried -- a completed kernel whose release
    is incomplete is a real bug and aborts immediately.
    """
    last_exc: StageFailure | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _run_stage(
                api, owner, "export", runs_dir, timeouts, poll_interval,
                keep_push_dir,
                extra_datasets=(checkpoint_ref,),
                download_map=download_map,
            )
        except StageFailure as exc:
            last_exc = exc
            print(f"[export] attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                wait = retry_delay * attempt
                print(f"[export] waiting {wait:.0f}s for input-mirror sync, "
                      f"then retrying ...")
                time.sleep(wait)
    raise StageFailure(
        f"export failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


def _corpus_accepted(corpus_report: dict, pipeline_corpus: dict) -> int:
    if isinstance(pipeline_corpus, dict) and pipeline_corpus.get("accepted") is not None:
        return int(pipeline_corpus["accepted"])
    return int(corpus_report.get("accepted") or 0)


def _corpus_breakdown(corpus_report: dict) -> str:
    parts = [
        f"{k}={corpus_report.get(k)}"
        for k in ("total", "accepted", "rejected", "errors", "acceptance_rate")
        if corpus_report.get(k) is not None
    ]
    return ", ".join(parts) if parts else repr(corpus_report)[:200]


def _year_range(corpus_report: dict) -> tuple:
    years = [
        s["year"]
        for s in corpus_report.get("samples", [])
        if isinstance(s, dict)
        and s.get("status") == "accepted"
        and s.get("year") is not None
    ]
    if not years:
        plan = corpus_report.get("plan") or {}
        years = plan.get("years") or corpus_report.get("config", {}).get("years") or []
    return (min(years), max(years)) if years else (None, None)


def _check_system_check(stage: dict) -> dict:
    validation = _load_json(stage["reports"].get("validation"), "validation.json")
    if not validation.get("passed"):
        raise StageFailure(
            "system check FAILED: " + _describe_issues(validation.get("issues", []))
        )
    errors = [i for i in validation.get("issues", []) if i.get("severity") == "error"]
    if errors:
        raise StageFailure(
            "system check passed but reports error-severity issues: "
            + _describe_issues(errors)
        )
    stage["validation"] = validation
    return stage


def _check_train(stage: dict) -> dict:
    pipeline = _load_json(stage["reports"].get("pipeline"), "pipeline.json")
    corpus_report = _load_json(stage["reports"].get("corpus"), "corpus.json")

    training = pipeline.get("training", {})
    if training.get("status") != "completed":
        raise StageFailure(
            "training did not complete: status="
            + repr(training.get("status"))
            + " reason="
            + str(training.get("reason", "(none)"))
        )
    accepted = _corpus_accepted(corpus_report, pipeline.get("corpus", {}))
    if accepted <= 0:
        raise StageFailure(
            "corpus has 0 accepted observations - nothing to train or export. "
            "Breakdown: " + _corpus_breakdown(corpus_report)
        )

    ckpt_info = _load_json(stage["reports"].get("checkpoint.json"), "checkpoint.json")
    if not ckpt_info.get("found"):
        raise StageFailure(
            "train finished but no checkpoint was registered (checkpoint.json found=false)"
        )
    ckpt_rel = ckpt_info.get("repo_relative")
    if not ckpt_rel:
        raise StageFailure("checkpoint.json is missing repo_relative")
    ckpt_dl = download_paths(
        stage["kernel_ref"],
        {"checkpoint.pt": "CropPrep/" + ckpt_rel},
        stage["run_dir"] / "checkpoint",
    )
    ckpt_pt = ckpt_dl.get("checkpoint.pt")
    if not ckpt_pt:
        raise StageFailure(
            f"checkpoint file missing in train kernel output: CropPrep/{ckpt_rel}"
        )

    stage["pipeline"] = pipeline
    stage["corpus"] = corpus_report
    stage["checkpoint"] = {
        "pt": str(ckpt_pt),
        "repo_relative": ckpt_rel,
        "run_dir": ckpt_info.get("run_dir"),
        "registered": ckpt_info.get("registered"),
    }

    sources = download_paths(
        stage["kernel_ref"],
        RELEASE_SOURCES_PATHS,
        stage["run_dir"] / "release_sources",
    )
    stage["release_sources"] = {
        name: (str(path) if path else None)
        for name, path in sources.items()
    }
    present = [name for name, path in sources.items() if path]
    print(f"[train] release sources present: {len(present)}/{len(sources)}")
    if "preprocess/scaler.pkl" not in sources or not sources["preprocess/scaler.pkl"]:
        print("[train] WARNING scaler.pkl missing - release package will be partial")
    return stage


def publish_checkpoint(
    api: KaggleApi,
    owner: str,
    ckpt_pt: Path,
    version_notes: str,
    dataset_slug: str,
    *,
    sources_dir: Path | None = None,
    poll_timeout: float = 600.0,
    poll_interval: float = 10.0,
) -> str:
    """Upload ``ckpt_pt`` (+ train-side release sources) as a new dataset version.

    ``sources_dir`` (the ``release_sources/`` tree downloaded from the train
    kernel) is copied into the upload folder under ``release_sources/`` so the
    export kernel mounts it at ``/kaggle/input/<dataset>/release_sources/``.
    Falls back to creating the (private) dataset when no version exists yet,
    then polls ``dataset_status`` until the dataset is ``complete``.
    """
    ref = f"{owner}/{dataset_slug}"
    with tempfile.TemporaryDirectory(prefix="cropfusion_ckpt_") as tmp:
        folder = Path(tmp)
        shutil.copy2(ckpt_pt, folder / "checkpoint.pt")
        if sources_dir is not None and sources_dir.exists():
            shutil.copytree(sources_dir, folder / "release_sources")
            print(f"[dataset] staged release_sources from {sources_dir}")
        (folder / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "id": ref,
                    "title": "CropFusion Checkpoints",
                    "subtitle": "Latest trained checkpoint + release sources "
                                "for the export kernel",
                    "licenses": [{"name": "other"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            api.dataset_create_version(str(folder), version_notes=version_notes, quiet=True)
            print(f"[dataset] uploaded new version of {ref}")
        except Exception as exc_version:  # noqa: BLE001 - fall back to create
            try:
                api.dataset_create_new(str(folder), public=False, quiet=True)
                print(f"[dataset] created new dataset {ref} "
                      f"(version attempt failed: {exc_version})")
            except Exception as exc_new:  # noqa: BLE001 - report both failures
                raise StageFailure(
                    "checkpoint dataset handoff failed "
                    f"(version: {exc_version}; create: {exc_new})"
                ) from exc_new

    deadline = time.monotonic() + poll_timeout
    status = None
    while time.monotonic() < deadline:
        try:
            status = str(api.dataset_status(ref)).lower()
        except Exception:  # noqa: BLE001 - transient status API errors
            time.sleep(5)
            continue
        if status in ("complete", "ready"):
            print(f"[dataset] {ref} ready (status={status})")
            return ref
        if status in (
            "failed", "error", "canceled", "cancelled", "cancel_requested",
            "deleted",
        ):
            raise StageFailure(f"checkpoint dataset {ref} ended in status {status}")
        time.sleep(poll_interval)
    raise StageFailure(
        f"checkpoint dataset {ref} not ready within {poll_timeout:g}s "
        f"(last status {status})"
    )


def _wait_dataset_file(
    api: KaggleApi,
    ref: str,
    filename: str,
    *,
    timeout: float,
    interval: float = 15.0,
) -> None:
    """Wait until ``filename`` is actually listed in dataset ``ref``.

    ``dataset_status == ready`` only means Kaggle accepted the upload; the
    input-mirror / catalog entry that the export kernel's ``/kaggle/input``
    mount reads can lag behind. Verify at the file level before pushing a
    consumer kernel so it never starts before the data is mountable.
    """
    deadline = time.monotonic() + timeout
    last: str | None = None
    while time.monotonic() < deadline:
        try:
            response = api.dataset_list_files(ref)
            names = {f.name for f in getattr(response, "dataset_files", [])}
            if filename in names:
                print(f"[dataset] {ref} lists {filename} (ready to mount)")
                return
            last = f"files={sorted(names)}"
        except Exception as exc:  # noqa: BLE001 - transient API errors while propagating
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(interval)
    raise StageFailure(
        f"dataset {ref} never exposed '{filename}' within {timeout:g}s (last: {last})"
    )


def _check_export(stage: dict) -> dict:
    release = stage["reports"]
    missing = [name for name, path in release.items() if not path]
    if missing:
        raise StageFailure(
            "export kernel finished but the release is incomplete; missing "
            "artifacts: " + ", ".join(missing)
        )
    manifest = _load_json(release["release.json"], "release.json")
    stage["manifest"] = manifest

    version = str(manifest.get("model_config", {}).get("version") or "")
    if not version:
        raise StageFailure("release.json has no model_config.version")
    package_paths = release_package_paths(version)
    package = download_paths(
        stage["kernel_ref"], package_paths, stage["run_dir"] / "release"
    )
    stage["release_package"] = {
        "version": version,
        "files": {
            name.removeprefix("release/"): (str(path) if path else None)
            for name, path in package.items()
        },
    }
    missing_package = [
        name.removeprefix("release/")
        for name, path in package.items()
        if not path
    ]
    if missing_package:
        raise StageFailure(
            f"release package (cropfusion_release-v{version}) is incomplete; "
            "missing files: " + ", ".join(missing_package)
        )
    print(f"[export] release package cropfusion_release-v{version} complete "
          f"({len(package)} files)")
    return stage


def _print_summary(
    stages: dict[str, dict],
    checkpoint_ref: str,
    runs_dir: Path,
    start_wall: float,
) -> None:
    total_s = time.monotonic() - start_wall
    sys_check = stages["system_check"].get("validation", {})
    train = stages["train"]
    corpus_report = train.get("corpus", {})
    years_lo, years_hi = _year_range(corpus_report)
    training = train.get("pipeline", {}).get("training", {})
    evaluation = (training.get("report") or {}).get("evaluation") or {}
    export = stages["export"]
    release = export["reports"]

    print("\n=============== CROPFUSION PIPELINE COMPLETE ===============")
    print(f"total wall-clock : {total_s / 60:.1f} min")
    for name in ("system_check", "train", "export"):
        st = stages[name]
        print(f"  {name:14s} {st['run_id']:22s} {st['status']:<10s} {st['url']}")
    print()
    print("system check     :", "passed" if sys_check.get("passed") else "FAILED",
          f"({sys_check.get('by_severity', {})})")
    print("corpus           :", _corpus_breakdown(corpus_report),
          f"| years {years_lo}-{years_hi}" if years_lo else "")
    print("training         :", training.get("status"),
          f"| accepted={training.get('accepted_observations')}")
    if evaluation:
        print("  evaluation     :", json.dumps(evaluation)[:300])
    ckpt = train["checkpoint"]
    print("checkpoint       :", ckpt["repo_relative"])
    print("  local          :", ckpt["pt"])
    print(f"checkpoint dataset: {checkpoint_ref} (handed to export kernel)")
    print("release          :", release["release.json"])
    print("  torchscript    :", release["model.torchscript.pt"])
    print("  onnx           :", release["model.onnx"])
    print("  model config   :", release["model.yaml"])
    release_package = export.get("release_package", {})
    if release_package.get("files"):
        print(f"release package  : cropfusion_release-v{release_package.get('version')}")
        pkg_dir = Path(
            next(iter(release_package["files"].values()))
        ).parent.parent.parent
        print("  local          :", pkg_dir)

    summary = {
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_clock_seconds": round(total_s, 1),
        "checkpoint_dataset": checkpoint_ref,
        "stages": {
            name: {
                "run_id": st["run_id"],
                "kernel": st["kernel_ref"],
                "kernel_version": st["version"],
                "kernel_status": st["status"],
                "kernel_url": st["url"],
                "run_dir": str(st["run_dir"]),
            }
            for name, st in stages.items()
        },
        "system_check": sys_check,
        "corpus": {
            "breakdown": _corpus_breakdown(corpus_report),
            "year_range": [years_lo, years_hi],
        },
        "training": {
            "status": training.get("status"),
            "accepted_observations": training.get("accepted_observations"),
            "evaluation": evaluation,
        },
        "checkpoint": ckpt,
        "release": {
            name: (str(path) if path else None)
            for name, path in release.items()
        },
        "release_package": export.get("release_package", {}),
    }
    out = runs_dir / f"pipeline-{_utc_stamp()}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\npipeline summary  :", out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the CropFusion pipeline end-to-end on Kaggle."
    )
    parser.add_argument(
        "--runs-dir",
        default="kaggle_runs",
        help="root directory for stage outputs (default: ./kaggle_runs)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="seconds between kernel status polls (default: 30)",
    )
    parser.add_argument(
        "--keep-push-dir",
        action="store_true",
        help="keep temporary kernel-metadata.json staging folders",
    )
    parser.add_argument(
        "--checkpoint-dataset",
        default=CHECKPOINT_DATASET_SLUG,
        help=f"dataset slug for the checkpoint handoff (default: {CHECKPOINT_DATASET_SLUG})",
    )
    parser.add_argument(
        "--dataset-ready-timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="max seconds to wait for the checkpoint file to appear in the "
             "dataset before pushing the export kernel (default: 600)",
    )
    parser.add_argument(
        "--dataset-grace-period",
        type=float,
        default=180.0,
        metavar="SECONDS",
        help="extra seconds to sleep after the checkpoint is listed before "
             "pushing the export kernel, letting Kaggle's input mirror catch "
             "up with the dataset-file API (default: 180)",
    )
    parser.add_argument(
        "--export-max-attempts",
        type=int,
        default=3,
        metavar="N",
        help="max pushes of the export kernel on infra failure (default: 3)",
    )
    parser.add_argument(
        "--export-retry-delay",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help="cooldown between export kernel retries (default: 300)",
    )
    for name, default in STAGE_TIMEOUTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}-timeout",
            type=float,
            default=default,
            dest=f"{name}_timeout",
            metavar="SECONDS",
            help=f"max seconds to wait for the {name} kernel (default: {default:g})",
        )
    args = parser.parse_args(argv)

    timeouts = {
        name: getattr(args, f"{name}_timeout") for name in STAGE_TIMEOUTS
    }
    runs_dir = Path(args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    owner = kagglehub.whoami()["username"]
    api = KaggleApi()
    api.authenticate()

    start_wall = time.monotonic()
    stages: dict[str, dict] = {}
    try:
        stage = _run_stage(
            api, owner, "system_check", runs_dir, timeouts, args.poll_interval,
            args.keep_push_dir,
        )
        stages["system_check"] = _check_system_check(stage)

        stage = _run_stage(
            api, owner, "train", runs_dir, timeouts, args.poll_interval,
            args.keep_push_dir,
            download_map={**REPORT_PATHS, "checkpoint.json": CHECKPOINT_REL},
        )
        stages["train"] = _check_train(stage)

        ckpt_pt = Path(stages["train"]["checkpoint"]["pt"])
        sources_dir = stages["train"]["run_dir"] / "release_sources"
        version_notes = (
            f"handoff from cropfusion-train run {stages['train']['run_id']} "
            f"({datetime.now(UTC).isoformat()})"
        )
        checkpoint_ref = publish_checkpoint(
            api, owner, ckpt_pt, version_notes, args.checkpoint_dataset,
            sources_dir=sources_dir,
        )
        # The input mirror can lag the dataset-status API; only push the export
        # kernel once the checkpoint is actually listed (mountable), then give
        # the mirror extra time to catch up before the kernel snapshots it.
        _wait_dataset_file(
            api, checkpoint_ref, "checkpoint.pt", timeout=args.dataset_ready_timeout
        )
        if args.dataset_grace_period > 0:
            print(f"[dataset] grace period {args.dataset_grace_period:.0f}s "
                  f"for input-mirror sync ...")
            time.sleep(args.dataset_grace_period)

        stage = _run_export_with_retry(
            api, owner, runs_dir, timeouts, args.poll_interval,
            args.keep_push_dir, checkpoint_ref, RELEASE_PATHS,
            max_attempts=args.export_max_attempts,
            retry_delay=args.export_retry_delay,
        )
        stages["export"] = _check_export(stage)
    except StageFailure as exc:
        print(f"\n[FAIL] pipeline aborted: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[ABORT] interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - unexpected infrastructure error
        print(f"\n[FAIL] unexpected error: {exc}", file=sys.stderr)
        return 2

    _print_summary(stages, checkpoint_ref, runs_dir, start_wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
