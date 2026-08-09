#!/usr/bin/env python3
"""Run a CropPrep Kaggle notebook end-to-end and fetch its infrastructure reports.

Pushes ``training/kaggle/notebooks/<name>.ipynb`` to Kaggle as a private GPU
kernel, waits for the run to finish, then downloads the three infrastructure
reports (``pipeline.json``, ``corpus.json``, ``validation.json``) into
``./kaggle_runs/<run_id>/`` so a human or a subsequent script can review them
without the Kaggle web UI.

API surface used
----------------
- kagglehub 1.0.2 exposes *notebook output download* and identity only::

      kagglehub.whoami()                                              # -> owner slug
      kagglehub.notebook_output_download(handle, path=..., ...)       # -> per-file output download

  It has NO kernel-push / status / logs API in this installed version, so those
  three steps go through the ``kaggle`` package (reads the same configured
  credentials, ``~/.kaggle/access_token``)::

      KaggleApi().authenticate()
      api.kernels_push(push_dir)       # -> ApiSaveKernelResponse(version_number, url, ...)
      api.kernels_status(kernel)       # -> status (KernelWorkerStatus enum)
      api.kernels_logs(kernel)         # -> session log text

This driver only pushes the selected notebook and its kernel-metadata.json; it
does not change sharing/visibility or touch any other kernel.

Usage::

    python scripts/run_kaggle_notebook.py --notebook system_check
    python scripts/run_kaggle_notebook.py --notebook train --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
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

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_SOURCE = "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"

NOTEBOOKS = {
    "system_check": {
        "slug": "cropfusion-system-check",
        "title": "CropFusion System Check",
        "file": "system_check.ipynb",
    },
    "train": {
        "slug": "cropfusion-train",
        "title": "CropFusion Train",
        "file": "train.ipynb",
    },
    "evaluate": {
        "slug": "cropfusion-evaluate",
        "title": "CropFusion Evaluate",
        "file": "evaluate.ipynb",
    },
    "export": {
        "slug": "cropfusion-export",
        "title": "CropFusion Export",
        "file": "export.ipynb",
    },
}

# Kernel output is rooted at /kaggle/working; the repo lives in CropPrep/, so
# reports appear under the CropPrep/ prefix inside the output bundle.
REPORT_PATHS = {
    "pipeline": "CropPrep/training/kaggle/outputs/reports/pipeline.json",
    "corpus": "CropPrep/training/kaggle/outputs/reports/corpus.json",
    "validation": "CropPrep/training/kaggle/outputs/reports/validation.json",
}

TERMINAL_OK = {"COMPLETE"}
TERMINAL_FAIL = {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}


def build_push_dir(
    notebook: dict,
    owner: str,
    *,
    extra_dataset_sources: Sequence[str] = (),
    keep: bool = False,
) -> Path:
    """Write kernel-metadata.json + the notebook into a clean temp folder.

    The push folder must contain nothing else, or Kaggle uploads those files
    too. ``extra_dataset_sources`` are added alongside the primary imagery
    dataset (e.g. a private checkpoint dataset for the export notebook).
    """
    src = REPO_ROOT / "training" / "kaggle" / "notebooks" / notebook["file"]
    if not src.exists():
        raise FileNotFoundError(f"notebook not found: {src}")

    push_dir = Path(tempfile.mkdtemp(prefix="kaggle_push_"))
    shutil.copy2(src, push_dir / notebook["file"])

    dataset_sources = [DATASET_SOURCE, *extra_dataset_sources]
    metadata = {
        "id": f"{owner}/{notebook['slug']}",
        "title": notebook["title"],
        "code_file": notebook["file"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[push] staged {notebook['file']} + kernel-metadata.json in {push_dir}")
    if not keep:
        import atexit

        atexit.register(shutil.rmtree, push_dir, ignore_errors=True)
    return push_dir


def push_notebook(api: KaggleApi, push_dir: Path) -> tuple[int | None, str]:
    resp = api.kernels_push(str(push_dir))
    if getattr(resp, "error", None):
        raise RuntimeError(f"Kaggle rejected kernel push: {resp.error}")
    invalid = [
        name
        for name in (
            "invalid_dataset_sources",
            "invalid_kernel_sources",
            "invalid_competition_sources",
            "invalid_model_sources",
            "invalid_tags",
        )
        if getattr(resp, name, None)
    ]
    if invalid:
        raise RuntimeError(f"Kaggle rejected kernel push (invalid sources): {invalid}")
    return resp.version_number, resp.url


def wait_for_completion(
    api: KaggleApi, kernel_ref: str, timeout: float, poll_interval: float
) -> str:
    deadline = time.monotonic() + timeout
    last = None
    while True:
        resp = api.kernels_status(kernel_ref)
        status = getattr(resp, "status", None)
        name = getattr(status, "name", status)
        last = name
        if name in TERMINAL_OK:
            return name
        if name in TERMINAL_FAIL:
            message = getattr(resp, "failure_message", "") or ""
            raise RuntimeError(
                f"Kaggle kernel run failed with status '{name}': {message}".strip()
            )
        print(f"[poll] {kernel_ref} status = {name} ...")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"kernel {kernel_ref} did not finish within {timeout:.0f}s "
                f"(last status: {last})"
            )
        time.sleep(poll_interval)


def download_paths(
    kernel_ref: str, rel_paths: dict[str, str], run_dir: Path
) -> dict[str, Path | None]:
    """Download each ``name -> kernel-relative path``; missing files (404) are
    recorded as ``None`` so callers can distinguish "not produced" from a hard
    download error (which still raises)."""
    results: dict[str, Path | None] = {}
    for name, rel in rel_paths.items():
        try:
            path = kagglehub.notebook_output_download(
                kernel_ref,
                path=rel,
                force_download=True,
                output_dir=str(run_dir),
            )
            results[name] = Path(path)
            print(f"[download] {name} -> {path}")
        except Exception as exc:  # noqa: BLE001 - missing reports are tolerated
            print(f"[download] {name} not present ({type(exc).__name__}): {exc}")
            results[name] = None
    return results


def download_reports(kernel_ref: str, run_dir: Path) -> dict[str, Path | None]:
    """Download the three infrastructure reports."""
    return download_paths(kernel_ref, REPORT_PATHS, run_dir)


def parse_log(log_text: str) -> dict:
    info: dict = {}
    m = re.search(r"system_check_exit=(\d+)", log_text)
    if m:
        info["system_check_exit"] = int(m.group(1))
    m = re.search(r"\bpassed:\s*(True|False)", log_text)
    if m:
        info["passed"] = m.group(1) == "True"
    m = re.search(r"->\s*(READY|NOT READY)", log_text)
    if m:
        info["bootstrap"] = m.group(1)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a CropPrep Kaggle notebook end-to-end and fetch reports."
    )
    parser.add_argument(
        "--notebook",
        choices=sorted(NOTEBOOKS),
        default="system_check",
        help="which notebook to push + run (default: system_check)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2700.0,
        help="max seconds to wait for the kernel run (default: 2700)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="seconds between status polls (default: 30)",
    )
    parser.add_argument(
        "--runs-dir",
        default="kaggle_runs",
        help="root directory for run outputs (default: ./kaggle_runs)",
    )
    parser.add_argument(
        "--keep-push-dir",
        action="store_true",
        help="keep the temporary kernel-metadata.json staging folder",
    )
    args = parser.parse_args()

    notebook = NOTEBOOKS[args.notebook]
    owner = kagglehub.whoami()["username"]
    kernel_ref = f"{owner}/{notebook['slug']}"

    api = KaggleApi()
    api.authenticate()

    push_dir = build_push_dir(notebook, owner, keep=args.keep_push_dir)
    version, url = push_notebook(api, push_dir)
    slug_base = notebook["slug"].replace("cropfusion-", "")
    if version:
        run_id = f"{slug_base}-v{version}"
        version_label = str(version)
    else:
        version_label = "new"
        run_id = f"{slug_base}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    print(f"[push] {kernel_ref} version {version_label} -> {url}")

    status = wait_for_completion(api, kernel_ref, args.timeout, args.poll_interval)

    run_dir = Path(args.runs_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reports = download_reports(kernel_ref, run_dir)

    log_text = api.kernels_logs(kernel_ref)
    log_info = parse_log(log_text)
    log_path = run_dir / "kernel.log"
    log_path.write_text(log_text, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "notebook": args.notebook,
        "kernel": kernel_ref,
        "kernel_version": version,
        "kernel_url": url,
        "kernel_status": status,
        "log": log_info,
        "reports": {
            name: (str(p) if p else None) for name, p in reports.items()
        },
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("=== CropFusion Kaggle run summary ===")
    print(f"notebook      : {args.notebook}")
    print(f"kernel        : {kernel_ref} (v{version_label})")
    print(f"run id        : {run_id}")
    print(f"kernel status : {status}")
    if "system_check_exit" in log_info:
        print(f"system_check  : exit {log_info['system_check_exit']} "
              f"| passed {log_info.get('passed')}")
    if "bootstrap" in log_info:
        print(f"bootstrap     : {log_info['bootstrap']}")
    print("reports       :")
    for name in ("validation", "pipeline", "corpus"):
        p = reports.get(name)
        if p:
            print(f"  {name:12s} -> {p}")
        else:
            print(f"  {name:12s} -> (not produced by this notebook)")
    print(f"kernel log    : {log_path}")
    print(f"summary       : {run_dir / 'summary.json'}")
    print(f"kernel url    : {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
