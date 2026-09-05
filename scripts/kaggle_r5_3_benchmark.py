"""R5.3 — Automated Local -> Kaggle Benchmark-Optimization Training Deployment.

Local launcher that prepares, submits, monitors, and retrieves the CropFusion
R5.3 benchmark run from Kaggle GPU infrastructure. The training notebook runs
the real multimodal model on the **R5.2.9-enriched** corpus
(`crop_supervised_v2.csv` + `crop_supervised_v2.0_manifest.json`, 10,674
benchmark-eligible rows) on the frozen spatial leave-one-taluk-out split.

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

    python scripts/kaggle_r5_3_benchmark.py check
    python scripts/kaggle_r5_3_benchmark.py train --confirm
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

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SRC = REPO_ROOT / "training" / "kaggle" / "notebooks" / "R5_3_benchmark.ipynb"
DEPLOY_DIR = REPO_ROOT / "training" / "kaggle" / "deployment" / "r5_3_benchmark"
MANIFEST_PATH = REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"
FROZEN_CSV = REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "r5_3_benchmark"

EXPECTED_TOTAL = 10_674
EXPECTED_TRAIN = 5_924
EXPECTED_VAL = 2_459
EXPECTED_TEST = 2_291

DATASET_ID = "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"
KERNEL_SLUG = "r5-3-cropfusion-benchmark"

DEFAULT_POLL_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 14_400  # 4 hours


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], check: bool = True, **kw: Any) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        env=env,
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
    username = _get_kaggle_username()
    if not username:
        return None
    return f"{username}/{KERNEL_SLUG}"


# ---------------------------------------------------------------------------
# CHECK
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    _print_section("CHECKING PREREQUISITES")
    ok = True

    creds = Path(os.path.expanduser("~")) / ".kaggle" / "kaggle.json"
    _print_row("Kaggle credentials", "PASS" if creds.exists() else "MISSING")
    if not creds.exists():
        ok = False

    try:
        import kaggle  # noqa: F401
        _print_row("kaggle module", "PASS")
    except Exception:
        _print_row("kaggle module", "MISSING")
        ok = False

    for path, label in [
        (NOTEBOOK_SRC, "R5.3 notebook"),
        (MANIFEST_PATH, "v2 manifest"),
        (FROZEN_CSV, "v2 CSV"),
    ]:
        found = path.exists()
        _print_row(label, "PASS" if found else "MISSING")
        if not found:
            ok = False
        else:
            _print_row(f"{label} sha256", _sha256(path))

    return 0 if ok else 1


# ---------------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------------


def _inject_epochs_cell(notebook_path: Path, epochs: int) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = f"import os\nos.environ['R5_3_EPOCHS'] = '{epochs}'\n"
    cell = {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}
    # Insert a runnable epoch-pinning cell just before the Full Pipeline cell.
    insert_before = None
    for idx, c in enumerate(nb["cells"]):
        if "run_pipeline.py" in "".join(c.get("source", [])):
            insert_before = idx
            break
    assert insert_before is not None
    nb["cells"].insert(insert_before, cell)
    notebook_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def cmd_prepare(args: argparse.Namespace) -> int:
    _print_section("PREPARING DEPLOYMENT DIRECTORY")
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    dst_nb = DEPLOY_DIR / "R5_3_benchmark.ipynb"
    shutil.copy2(NOTEBOOK_SRC, dst_nb)
    _print_row("Notebook", f"Copied -> {dst_nb.name}")

    test_epochs = getattr(args, "test_epochs", None)
    if test_epochs is not None:
        _inject_epochs_cell(dst_nb, int(test_epochs))
        _print_row("Test epochs", f"{test_epochs} (injected R5_3_EPOCHS)")
    else:
        _print_row("Test epochs", "off (FULL 30-EPOCH RUN)")

    username = _get_kaggle_username()
    if not username:
        print("  [ERROR] Cannot determine Kaggle username.")
        return 1
    kernel_id = f"{username}/{KERNEL_SLUG}"

    metadata = {
        "id": kernel_id,
        "title": "R5.3 CropFusion Benchmark Optimization",
        "code_file": "R5_3_benchmark.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET_ID],
    }
    (DEPLOY_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _print_row("Kernel ID", kernel_id)
    _print_row("Dataset", DATASET_ID)
    _print_row("GPU", "enabled")

    _print_section("DEPLOYMENT VERIFICATION")
    for fname in ["R5_3_benchmark.ipynb", "kernel-metadata.json"]:
        fp = DEPLOY_DIR / fname
        _print_row(fname, f"{'PASS' if fp.exists() else 'FAIL'}  ({fp.stat().st_size:,} bytes)")

    print("\n  DEPLOYMENT READY")
    return 0


# ---------------------------------------------------------------------------
# PUSH / RUN
# ---------------------------------------------------------------------------


def cmd_push(args: argparse.Namespace) -> int:
    _print_section("PUSHING TO KAGGLE")
    if not DEPLOY_DIR.exists():
        print("  [ERROR] Deployment directory not found. Run 'prepare' first.")
        return 1
    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] Cannot determine kernel ID. Run 'check' first.")
        return 1
    _print_row("Kernel", kernel_id)
    cmd = ["kaggle", "kernels", "push", "-p", str(DEPLOY_DIR)]
    if args.timeout:
        cmd.extend(["-t", str(args.timeout)])
    result = _run(cmd, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode == 0 and "Kernel push error" not in result.stdout:
        _print_row("Status", "PUSHED")
        return 0
    _print_row("Status", f"FAILED (exit code {result.returncode})")
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    _print_section("STARTING REMOTE EXECUTION")
    push_result = cmd_push(args)
    if push_result != 0:
        return push_result
    print("\n  Remote execution started (push triggers auto-run on Kaggle).")
    return 0


# ---------------------------------------------------------------------------
# STATUS / OUTPUT / TRAIN
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] Cannot determine kernel ID.")
        return 1
    result = _kaggle("kernels", "status", kernel_id, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def cmd_output(args: argparse.Namespace) -> int:
    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] Cannot determine kernel ID.")
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _kaggle("kernels", "output", kernel_id, "-p", str(OUTPUT_DIR), check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    _print_row("Output dir", str(OUTPUT_DIR))
    return 0 if result.returncode == 0 else 1


def cmd_train(args: argparse.Namespace) -> int:
    if not getattr(args, "confirm", False):
        print("  [ERROR] train requires --confirm (safety gate).")
        return 1
    if cmd_prepare(args) != 0:
        return 1
    if cmd_run(args) != 0:
        return 1
    print("\n  Training launched. Monitoring...")

    kernel_id = _get_kernel_id()
    assert kernel_id
    timeout = args.timeout or DEFAULT_TIMEOUT_SECONDS
    poll = args.poll_seconds or DEFAULT_POLL_SECONDS
    start = time.time()
    while time.time() - start < timeout:
        result = _kaggle("kernels", "status", kernel_id, check=False)
        out = result.stdout
        if "complete" in out.lower():
            print(out)
            print("\n  RUN COMPLETE")
            return cmd_output(args)
        if "error" in out.lower() or "cancelled" in out.lower():
            print(out)
            print("\n  RUN FAILED")
            return 1
        print(f"  [{time.time() - start:.0f}s] {out.strip() or 'pending...'}")
        time.sleep(poll)
    print("\n  TIMED OUT monitoring (check 'status' / 'output').")
    return 1


# ---------------------------------------------------------------------------
# VERIFY OUTPUT
# ---------------------------------------------------------------------------


def cmd_verify_output(args: argparse.Namespace) -> int:
    _print_section("VERIFYING DOWNLOADED OUTPUT")
    pipeline = OUTPUT_DIR / "training" / "kaggle" / "outputs" / "reports" / "pipeline.json"
    candidates = [
        pipeline,
        OUTPUT_DIR / "pipeline.json",
        OUTPUT_DIR / "output" / "training" / "kaggle" / "outputs" / "reports" / "pipeline.json",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        print("  pipeline.json not found under", OUTPUT_DIR)
        print("  Files present:")
        for p in sorted(OUTPUT_DIR.rglob("pipeline.json")):
            print("   ", p)
        return 1
    report = json.loads(found[0].read_text(encoding="utf-8"))
    training = report.get("training", {})
    print("  Status:", training.get("status", "unknown"))
    print("  Corpus:", report.get("corpus", {}).get("total"))
    print("  Train/Val/Test:", report.get("corpus", {}).get("train"),
          "/", report.get("corpus", {}).get("val"),
          "/", report.get("corpus", {}).get("test"))
    if training.get("status") == "completed":
        metrics = training.get("report", {}).get("evaluation", {}).get("metrics", {})
        print("  Test accuracy:", metrics.get("crop/accuracy"))
        print("  Test macro F1:", metrics.get("crop/macro_f1"))
        print("  Test weighted F1:", metrics.get("crop/weighted_f1"))
        return 0
    print("  Training did not complete; inspect logs.")
    return 1


# ---------------------------------------------------------------------------
# FULL
# ---------------------------------------------------------------------------


def cmd_full(args: argparse.Namespace) -> int:
    # prepare -> push -> run (trigger) -> monitor (train-style wait) -> output
    # -> verify. We do NOT call cmd_train again (it re-runs prepare/push).
    from functools import partial

    _print_section("STEP CHECK")
    if cmd_check(args) != 0:
        return 1
    _print_section("STEP PREPARE")
    if cmd_prepare(args) != 0:
        return 1
    _print_section("STEP PUSH")
    if cmd_push(args) != 0:
        return 1
    _print_section("STEP RUN (TRIGGER)")
    if cmd_run(args) != 0:
        return 1

    # Monitor for completion, then download and verify.
    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] cannot determine kernel id")
        return 1
    timeout = args.timeout or DEFAULT_TIMEOUT_SECONDS
    poll = args.poll_seconds or DEFAULT_POLL_SECONDS
    start = time.time()
    status_pending = True
    while time.time() - start < timeout:
        result = _kaggle("kernels", "status", kernel_id, check=False)
        out = result.stdout
        if "complete" in out.lower():
            status_pending = False
            break
        if "error" in out.lower() or "cancelled" in out.lower():
            print(out)
            print("\n  RUN FAILED")
            return 1
        print(f"  [{time.time() - start:.0f}s] {out.strip() or 'pending...'}")
        time.sleep(poll)
    if status_pending:
        print("\n  TIMED OUT monitoring (run was triggered; check 'status' / 'output').")
        return 1

    _print_section("STEP OUTPUT")
    if cmd_output(args) != 0:
        return 1
    _print_section("STEP VERIFY-OUTPUT")
    if cmd_verify_output(args) != 0:
        return 1
    print("\n  FULL PIPELINE COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-r5-3-benchmark")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Local prerequisites only")
    p_prepare = sub.add_parser("prepare", help="Prepare the Kaggle deployment directory")
    p_prepare.add_argument("--test-epochs", type=int, default=None,
                           help="Short run: inject R5_3_EPOCHS=<n> into the deployed notebook")
    sub.add_parser("push", help="Push/update notebook to Kaggle")
    sub.add_parser("run", help="Start remote execution")
    sub.add_parser("status", help="Show current Kaggle status")
    sub.add_parser("output", help="Download latest output")

    p_train = sub.add_parser("train", help="Push + execute + monitor")
    p_train.add_argument("--confirm", action="store_true")
    p_train.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p_train.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    p_train.add_argument("--test-epochs", type=int, default=None)

    sub.add_parser("verify-output", help="Validate downloaded artifacts")

    p_full = sub.add_parser("full", help="check -> prepare -> push -> run -> monitor -> download -> verify")
    p_full.add_argument("--confirm", action="store_true")
    p_full.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p_full.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)

    args = parser.parse_args(argv)
    handlers = {
        "check": cmd_check, "prepare": cmd_prepare, "push": cmd_push,
        "run": cmd_run, "status": cmd_status, "output": cmd_output,
        "train": cmd_train, "verify-output": cmd_verify_output, "full": cmd_full,
    }
    fn = handlers.get(args.command)
    if fn is None:
        parser.print_help()
        return 2
    return fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())