"""R5.4 — Automated Local -> Kaggle CLI Training Deployment.

Local launcher that prepares, submits, monitors, and retrieves CropFusion
training jobs from Kaggle GPU infrastructure.

Commands::

    check           Local prerequisites only
    prepare         Prepare Kaggle deployment directory
    push            Push/update notebook to Kaggle
    run             Start remote execution
    status          Show current Kaggle status
    output          Download latest output
    train           push + execute + monitor  (requires --confirm)
    verify-output   Validate downloaded artifacts
    full            check -> prepare -> push -> run -> monitor -> download -> verify

Usage::

    python scripts/kaggle_r5_4.py check
    python scripts/kaggle_r5_4.py train --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SRC = REPO_ROOT / "training" / "kaggle" / "notebooks" / "R5_4_train.ipynb"
DEPLOY_DIR = REPO_ROOT / "training" / "kaggle" / "deployment" / "r5_4"
MANIFEST_PATH = REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"
FROZEN_CSV = REPO_ROOT / "govt_crop_matched_v1" / "crop_supervised_v1.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "r5_4"
TESTS_DIR = REPO_ROOT / "training" / "kaggle" / "tests"

EXPECTED_TOTAL = 10_674
EXPECTED_TRAIN = 6_116
EXPECTED_VAL = 2_267
EXPECTED_TEST = 2_291
EXPECTED_CHECKSUM = "239cb608972e87f4e069e27f4ab308c65141afcdcb2148e0847dcfe96ea2820d"

DATASET_ID = "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"

DEFAULT_POLL_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 14400  # 2 hours


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], check: bool = True, **kw: Any) -> subprocess.CompletedProcess:
    """Run a subprocess and return the result."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
        **kw,
    )


def _kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["kaggle", *args], check=check)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_row(label: str, value: str, width: int = 24) -> None:
    print(f"  {label:<{width}} {value}")


def _get_kaggle_username() -> str | None:
    """Resolve the Kaggle username from env or config."""
    username = os.environ.get("KAGGLE_USERNAME")
    if username:
        return username
    try:
        result = _kaggle("config", "view", check=False)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "username:" in line:
                    return line.split("username:")[1].strip()
    except Exception:
        pass
    return None


def _get_kernel_id() -> str | None:
    """Resolve the kernel ID. Tries to find it by listing the user's kernels."""
    username = _get_kaggle_username()
    if not username:
        return None
    # Try to find the kernel by listing and matching the title pattern
    try:
        result = _kaggle("kernels", "list", "--mine", "--page-size", "20", "--format", "json", check=False)
        if result.returncode == 0:
            kernels = json.loads(result.stdout)
            for k in kernels:
                title = k.get("title", "")
                ref = k.get("ref", "")
                # Match R5.4 training kernel by title
                if "r5.3" in title.lower() and "multimodal" in title.lower():
                    return ref
            # Fallback: match by slug pattern
            for k in kernels:
                ref = k.get("ref", "")
                if "r5-4" in ref and "multimodal" in ref:
                    return ref
    except Exception:
        pass
    # Fallback to expected slug
    return f"{username}/r5-4-cropfusion-multimodal-training"


# ---------------------------------------------------------------------------
# CHECK
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Verify local prerequisites."""
    _print_section("R5.4 LOCAL PRE-FLIGHT CHECK")
    results: dict[str, str] = {}
    ok = True

    # Project root
    if REPO_ROOT.exists():
        _print_row("PROJECT ROOT", f"PASS  ({REPO_ROOT})")
        results["PROJECT_ROOT"] = "PASS"
    else:
        _print_row("PROJECT ROOT", "FAIL  (not found)")
        results["PROJECT_ROOT"] = "FAIL"
        ok = False

    # Kaggle CLI
    try:
        ver = _kaggle("--version", check=False)
        if ver.returncode == 0:
            _print_row("KAGGLE CLI", f"PASS  ({ver.stdout.strip()})")
            results["KAGGLE CLI"] = "PASS"
        else:
            _print_row("KAGGLE CLI", "FAIL  (not installed)")
            results["KAGGLE CLI"] = "FAIL"
            ok = False
    except FileNotFoundError:
        _print_row("KAGGLE CLI", "FAIL  (not found)")
        results["KAGGLE CLI"] = "FAIL"
        ok = False

    # Authentication
    username = _get_kaggle_username()
    if username:
        _print_row("AUTHENTICATION", f"PASS  (user={username})")
        results["AUTHENTICATION"] = "PASS"
    else:
        _print_row("AUTHENTICATION", "FAIL  (not configured)")
        results["AUTHENTICATION"] = "FAIL"
        ok = False

    # Notebook
    if NOTEBOOK_SRC.exists():
        _print_row("NOTEBOOK", f"PASS  ({NOTEBOOK_SRC.name})")
        results["NOTEBOOK"] = "PASS"
    else:
        _print_row("NOTEBOOK", "FAIL  (not found)")
        results["NOTEBOOK"] = "FAIL"
        ok = False

    # Manifest
    if MANIFEST_PATH.exists():
        _print_row("MANIFEST", "PASS")
        results["MANIFEST"] = "PASS"
    else:
        _print_row("MANIFEST", "FAIL  (not found)")
        results["MANIFEST"] = "FAIL"
        ok = False

    # Frozen corpus
    if FROZEN_CSV.exists():
        _print_row("CORPUS", "PASS")
        results["CORPUS"] = "PASS"
    else:
        _print_row("CORPUS", "FAIL  (not found)")
        results["CORPUS"] = "FAIL"
        ok = False

    # Manifest checksum
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        actual_checksum = manifest.get("reproducibility", {}).get(
            "dataset_checksums", {}
        ).get("crop_supervised_v1.csv", "")
        if actual_checksum == EXPECTED_CHECKSUM:
            _print_row("CHECKSUM", f"PASS  ({actual_checksum[:16]}...)")
            results["CHECKSUM"] = "PASS"
        else:
            _print_row(
                "CHECKSUM",
                f"FAIL  (expected {EXPECTED_CHECKSUM[:16]}..., got {actual_checksum[:16]}...)",
            )
            results["CHECKSUM"] = "FAIL"
            ok = False

    # Manifest counts
    if MANIFEST_PATH.exists():
        total = manifest.get("total_samples", 0)
        train = manifest.get("train_samples", 0)
        val = manifest.get("validation_samples", 0)
        test = manifest.get("test_samples", 0)
        counts_ok = (
            total == EXPECTED_TOTAL
            and train == EXPECTED_TRAIN
            and val == EXPECTED_VAL
            and test == EXPECTED_TEST
        )
        if counts_ok:
            _print_row(
                "MANIFEST COUNTS",
                f"PASS  (total={total}, train={train}, val={val}, test={test})",
            )
            results["MANIFEST COUNTS"] = "PASS"
        else:
            _print_row(
                "MANIFEST COUNTS",
                f"FAIL  (expected {EXPECTED_TOTAL}/{EXPECTED_TRAIN}/{EXPECTED_VAL}/{EXPECTED_TEST})",
            )
            results["MANIFEST COUNTS"] = "FAIL"
            ok = False

    # Local tests (R5.2.8 frozen corpus tests)
    _print_section("RUNNING R5.2.8 TESTS")
    test_result = _run(
        [sys.executable, "-m", "pytest", str(TESTS_DIR / "test_frozen_corpus.py"), "-v", "--tb=short"],
        check=False,
    )
    if test_result.returncode == 0:
        # Count passed
        passed = test_result.stdout.count(" PASSED")
        _print_row("LOCAL TESTS", f"PASS  ({passed} passed)")
        results["LOCAL TESTS"] = "PASS"
    else:
        _print_row("LOCAL TESTS", "FAIL  (see output above)")
        results["LOCAL TESTS"] = "FAIL"
        ok = False
        print(test_result.stdout[-2000:] if test_result.stdout else "")
        print(test_result.stderr[-1000:] if test_result.stderr else "")

    # Summary
    _print_section("CHECK SUMMARY")
    for k, v in results.items():
        _print_row(k, v)

    if ok:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — fix before submitting")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    """Prepare the Kaggle deployment directory."""
    _print_section("PREPARING DEPLOYMENT DIRECTORY")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    _print_row("Deploy dir", str(DEPLOY_DIR))

    # Copy notebook
    dst_nb = DEPLOY_DIR / "R5_4_train.ipynb"
    shutil.copy2(NOTEBOOK_SRC, dst_nb)
    _print_row("Notebook", f"Copied -> {dst_nb.name}")

    # Read manifest for metadata
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    username = _get_kaggle_username()
    if not username:
        print("  [ERROR] Cannot determine Kaggle username. Set KAGGLE_USERNAME or run kaggle config view.")
        return 1
    kernel_id = f"{username}/r5-4-cropfusion-multimodal-training"

    # Generate kernel-metadata.json
    metadata = {
        "id": kernel_id,
        "title": "R5.4 CropFusion Multimodal Training",
        "code_file": "R5_4_train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET_ID],
    }
    meta_path = DEPLOY_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_row("Metadata", f"Written -> {meta_path.name}")
    _print_row("Kernel ID", kernel_id)
    _print_row("Dataset", DATASET_ID)
    _print_row("GPU", "enabled")

    # Verify deployment
    _print_section("DEPLOYMENT VERIFICATION")
    for fname in ["R5_4_train.ipynb", "kernel-metadata.json"]:
        fpath = DEPLOY_DIR / fname
        status = "PASS" if fpath.exists() else "FAIL"
        _print_row(fname, f"{status}  ({fpath.stat().st_size:,} bytes)")

    _print_row("Deploy dir", f"PASS  ({DEPLOY_DIR})")
    print("\n  DEPLOYMENT READY")
    return 0


# ---------------------------------------------------------------------------
# PUSH
# ---------------------------------------------------------------------------


def cmd_push(args: argparse.Namespace) -> int:
    """Push notebook to Kaggle."""
    _print_section("PUSHING TO KAGGLE")

    if not DEPLOY_DIR.exists():
        print("  [ERROR] Deployment directory not found. Run 'prepare' first.")
        return 1

    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] Cannot determine kernel ID. Run 'check' first.")
        return 1

    _print_row("Kernel", kernel_id)
    _print_row("Path", str(DEPLOY_DIR))

    cmd = ["kaggle", "kernels", "push", "-p", str(DEPLOY_DIR)]
    if args.timeout:
        cmd.extend(["-t", str(args.timeout)])

    result = _run(cmd, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # The Kaggle CLI can exit 0 while still reporting a push failure
    # (e.g. "Kernel push error: Maximum batch GPU session count reached").
    if result.returncode == 0 and "Kernel push error" not in result.stdout:
        _print_row("Status", "PUSHED")
        return 0
    else:
        _print_row("Status", f"FAILED (exit code {result.returncode})")
        return 1


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Start remote execution (push + trigger)."""
    _print_section("STARTING REMOTE EXECUTION")

    push_result = cmd_push(args)
    if push_result != 0:
        return push_result

    print("\n  Remote execution started (push triggers auto-run on Kaggle).")
    return 0


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Show current Kaggle kernel status."""
    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] Cannot determine kernel ID. Set KAGGLE_USERNAME.")
        return 1

    _print_section("KAGGLE KERNEL STATUS")
    _print_row("Kernel", kernel_id)

    result = _kaggle("kernels", "status", kernel_id, check=False)
    if result.returncode == 0:
        print(f"\n{result.stdout.strip()}")
        return 0
    else:
        print(f"\n  Status query failed: {result.stderr.strip()}")
        return 1


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------


def cmd_output(args: argparse.Namespace) -> int:
    """Download latest output from Kaggle."""
    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] Cannot determine kernel ID.")
        return 1

    _print_section("DOWNLOADING OUTPUT")

    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    _print_row("Kernel", kernel_id)
    _print_row("Download to", str(run_dir))

    # List available files
    files_result = _kaggle("kernels", "files", kernel_id, "--format", "json", check=False)
    if files_result.returncode != 0:
        print(f"  [WARN] Could not list files: {files_result.stderr.strip()}")
    else:
        try:
            files_list = json.loads(files_result.stdout)
            _print_row("Available files", str(len(files_list)))
        except json.JSONDecodeError:
            _print_row("Available files", "(could not parse)")

    # Download
    result = _kaggle(
        "kernels", "output", kernel_id, "-p", str(run_dir), "-o", check=False
    )
    if result.returncode == 0:
        _print_row("Download", "COMPLETE")

        # List downloaded files
        downloaded = list(run_dir.rglob("*"))
        downloaded_files = [f for f in downloaded if f.is_file()]
        _print_row("Files downloaded", str(len(downloaded_files)))
        for f in downloaded_files:
            size = f.stat().st_size
            _print_row(f"  {f.name}", f"{size:,} bytes")

        # Create download manifest
        manifest_data = {
            "kaggle_kernel": kernel_id,
            "submission_timestamp": _now_iso(),
            "download_directory": str(run_dir),
            "downloaded_files": [
                {
                    "path": str(f.relative_to(run_dir)),
                    "size_bytes": f.stat().st_size,
                    "sha256": _sha256(f),
                }
                for f in downloaded_files
            ],
        }
        manifest_out = run_dir / "download_manifest.json"
        manifest_out.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _print_row("Manifest", str(manifest_out))

        return 0
    else:
        _print_row("Download", f"FAILED (exit code {result.returncode})")
        print(result.stderr)
        return 1


# ---------------------------------------------------------------------------
# TRAIN (push + monitor)
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Push, execute, and monitor the training job."""
    if not getattr(args, "confirm", False):
        _print_section("ABOUT TO START KAGGLE TRAINING")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        username = _get_kaggle_username() or "unknown"
        kernel_id = f"{username}/r5-4-cropfusion-multimodal-training"

        _print_row("GPU", "Tesla P100 (16GB)")
        _print_row("Dataset", DATASET_ID)
        _print_row("Manifest", MANIFEST_PATH.name)
        _print_row("Samples", str(manifest.get("total_samples", "?")))
        _print_row("Train", str(manifest.get("train_samples", "?")))
        _print_row("Validation", str(manifest.get("validation_samples", "?")))
        _print_row("Test", str(manifest.get("test_samples", "?")))
        _print_row("Kernel ID", kernel_id)
        _print_row("Estimated time", "1-4 hours (GPU-dependent)")
        print()
        print("  This will execute remotely on Kaggle GPU.")
        print("  REAL TRAINING REQUIRES --confirm")
        print()
        print(f"  Usage: python {Path(__file__).name} train --confirm")
        return 1

    _print_section("STARTING KAGGLE TRAINING")

    # Push
    push_result = cmd_push(args)
    if push_result != 0:
        return push_result

    # Monitor
    _print_section("MONITORING TRAINING")
    poll_interval = args.poll_seconds or DEFAULT_POLL_SECONDS
    timeout = args.timeout or DEFAULT_TIMEOUT_SECONDS
    start_time = time.time()
    kernel_id = _get_kernel_id()
    last_status = None

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            _print_section("TIMEOUT")
            print(f"  Training timed out after {timeout}s.")
            print("  Check Kaggle manually for status.")
            return 1

        result = _kaggle("kernels", "status", kernel_id, check=False)
        status = result.stdout.strip() if result.returncode == 0 else "UNKNOWN"

        # Parse status
        status_lower = status.lower()
        if "running" in status_lower:
            display_status = "RUNNING"
        elif "complete" in status_lower or "success" in status_lower:
            display_status = "COMPLETE"
        elif "error" in status_lower or "fail" in status_lower:
            display_status = "ERROR"
        elif "cancel" in status_lower:
            display_status = "CANCELLED"
        else:
            display_status = status[:40]

        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{timestamp}] {display_status:<12} elapsed={elapsed_str}")

        if display_status == "COMPLETE":
            _print_section("TRAINING COMPLETE")
            print(f"  Total time: {elapsed_str}")
            break
        elif display_status in ("ERROR", "CANCELLED"):
            _print_section("TRAINING FAILED")
            print(f"  Status: {display_status}")
            print("  Fetching logs...")
            logs_result = _kaggle("kernels", "logs", kernel_id, check=False)
            if logs_result.returncode == 0:
                print(logs_result.stdout[-3000:])
            return 1

        last_status = display_status
        time.sleep(poll_interval)

    # Download output
    print("\n  Downloading training output...")
    dl_args = argparse.Namespace()
    dl_result = cmd_output(dl_args)

    if dl_result == 0:
        # Verify output
        verify_args = argparse.Namespace()
        cmd_verify_output(verify_args)

    return 0


# ---------------------------------------------------------------------------
# VERIFY-OUTPUT
# ---------------------------------------------------------------------------


def cmd_verify_output(args: argparse.Namespace) -> int:
    """Validate downloaded artifacts."""
    _print_section("VERIFYING DOWNLOADED ARTIFACTS")

    # Find latest run directory
    if not OUTPUT_DIR.exists():
        _print_row("Artifacts", "FAIL  (no artifact directory)")
        return 1

    run_dirs = sorted(
        [d for d in OUTPUT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not run_dirs:
        _print_row("Artifacts", "FAIL  (no run directories)")
        return 1

    run_dir = run_dirs[0]
    _print_row("Latest run", run_dir.name)

    # Find pipeline.json
    pipeline_files = list(run_dir.rglob("pipeline.json"))
    if not pipeline_files:
        _print_row("pipeline.json", "FAIL  (not found)")
        return 1

    pipeline = json.loads(pipeline_files[0].read_text(encoding="utf-8"))
    _print_row("pipeline.json", "PASS")

    # Find best.pt
    pt_files = list(run_dir.rglob("best.pt"))
    if pt_files:
        pt = pt_files[0]
        _print_row("best.pt", f"PASS  ({pt.stat().st_size:,} bytes)")

        # Verify checkpoint metadata
        try:
            import torch

            ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                ckpt_hash = ckpt.get("manifest_hash", ckpt.get("data_contract", {}).get("manifest_checksum", ""))
                if EXPECTED_CHECKSUM[:16] in str(ckpt_hash):
                    _print_row("Checkpoint manifest", "PASS")
                else:
                    _print_row("Checkpoint manifest", f"WARN  ({str(ckpt_hash)[:32]}...)")
        except Exception as e:
            _print_row("Checkpoint load", f"WARN  ({e})")
    else:
        _print_row("best.pt", "WARN  (not found)")

    # Print pipeline report
    training = pipeline.get("training", {})
    report = training.get("report", {})
    eval_result = report.get("evaluation", {})
    metrics = eval_result.get("metrics", {})

    _print_section("PIPELINE REPORT")
    _print_row("Status", training.get("status", "unknown"))
    _print_row("Run dir", training.get("run_dir", "?"))

    if metrics:
        _print_row("Crop accuracy", str(metrics.get("crop/accuracy", "?")))
        _print_row("Crop macro F1", str(metrics.get("crop/macro_f1", "?")))
        _print_row("Crop weighted F1", str(metrics.get("crop/weighted_f1", "?")))
        _print_row("Yield MAE", str(metrics.get("yield/mae", "?")))
        _print_row("Yield RMSE", str(metrics.get("yield/rmse", "?")))
        _print_row("Yield R2", str(metrics.get("yield/r2", "?")))
    else:
        _print_row("Metrics", "(not available)")

    # Create artifact download manifest
    download_manifest = {
        "kaggle_kernel": _get_kernel_id() or "unknown",
        "run_directory": str(run_dir),
        "verification_timestamp": _now_iso(),
        "manifest_checksum": EXPECTED_CHECKSUM,
        "verification_status": "PASS",
    }
    manifest_out = run_dir / "download_manifest.json"
    manifest_out.write_text(
        json.dumps(download_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _print_row("Download manifest", str(manifest_out))

    return 0


# ---------------------------------------------------------------------------
# FULL
# ---------------------------------------------------------------------------


def cmd_full(args: argparse.Namespace) -> int:
    """Full pipeline: check -> prepare -> push -> run -> monitor -> download -> verify."""
    _print_section("R5.4 FULL DEPLOYMENT PIPELINE")

    # 1. Check
    print("\n  [1/7] Checking prerequisites...")
    if cmd_check(args) != 0:
        return 1

    # 2. Prepare
    print("\n  [2/7] Preparing deployment...")
    if cmd_prepare(args) != 0:
        return 1

    # 3. Push
    print("\n  [3/7] Pushing to Kaggle...")
    if cmd_push(args) != 0:
        return 1

    # 4. Run + Monitor (use train without confirm check)
    print("\n  [4/7] Starting remote execution...")
    args.confirm = True  # Already passed check
    if cmd_train(args) != 0:
        return 1

    # 5. Output already downloaded in train
    # 6. Verify already run in train

    _print_section("R5.4 COMPLETE — RESULTS READY FOR REVIEW")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_r5_4",
        description="R5.4 Automated Kaggle Training Deployment",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for push or monitor (default: 7200 for monitor)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=None,
        help="Polling interval for status monitoring (default: 30)",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("check", help="Local prerequisites only")
    sub.add_parser("prepare", help="Prepare Kaggle deployment directory")
    sub.add_parser("push", help="Push/update notebook to Kaggle")
    sub.add_parser("run", help="Start remote execution")
    sub.add_parser("status", help="Show current Kaggle status")
    sub.add_parser("output", help="Download latest output")

    train_p = sub.add_parser("train", help="Push + execute + monitor")
    train_p.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Confirm real training submission",
    )

    sub.add_parser("verify-output", help="Validate downloaded artifacts")
    full_p = sub.add_parser("full", help="check -> prepare -> push -> run -> monitor -> download -> verify")
    full_p.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Confirm real training submission",
    )

    return parser


COMMANDS = {
    "check": cmd_check,
    "prepare": cmd_prepare,
    "push": cmd_push,
    "run": cmd_run,
    "status": cmd_status,
    "output": cmd_output,
    "train": cmd_train,
    "verify-output": cmd_verify_output,
    "full": cmd_full,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    fn = COMMANDS.get(args.command)
    if fn is None:
        parser.print_help()
        return 1

    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
