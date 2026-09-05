"""R5.5 classifier-collapse diagnostic — Kaggle GPU kernel launcher.

Prepares, pushes and monitors the R5.5 classifier-collapse diagnostic notebook
(``training/kaggle/notebooks/R5_5_diagnose_collapse.ipynb``), which runs
:mod:`training.kaggle.scripts.diagnose_collapse_kaggle` (GPU phases 3, 5, 9,
10, 11 and the 12-14 baselines) against the frozen R5.2.9-enriched corpus.

Subcommands (mirror ``scripts/kaggle_r5_3_benchmark.py`` conventions)::

    python scripts/kaggle_r5_5_diagnostic.py prepare
    python scripts/kaggle_r5_5_diagnostic.py push [--timeout N]
    python scripts/kaggle_r5_5_diagnostic.py status
    python scripts/kaggle_r5_5_diagnostic.py output
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SRC = (
    REPO_ROOT / "training" / "kaggle" / "notebooks" / "R5_5_diagnose_collapse.ipynb"
)
DEPLOY_DIR = REPO_ROOT / "training" / "kaggle" / "deployment" / "r5_5_diagnostic"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "r5_5_diagnostic"

DATASET_ID = "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"
KERNEL_SLUG = "r5-5-diagnose-collapse"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    try:
        result = _kaggle("config", "view", check=False)
        line = next(
            (ln for ln in result.stdout.splitlines() if ln.strip().startswith("- username:")),
            None,
        )
        if line:
            return line.split(":", 1)[1].strip()
    except Exception:
        pass
    local = Path.home() / ".kaggle" / "kaggle.json"
    if local.exists():
        try:
            return json.loads(local.read_text(encoding="utf-8")).get("username")
        except Exception:
            return None
    return None


def _get_kernel_id() -> str | None:
    username = _get_kaggle_username()
    return f"{username}/{KERNEL_SLUG}" if username else None


def cmd_prepare(args: argparse.Namespace) -> int:
    _print_section("PREPARING DEPLOYMENT DIRECTORY")
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    dst_nb = DEPLOY_DIR / NOTEBOOK_SRC.name
    shutil.copy2(NOTEBOOK_SRC, dst_nb)
    _print_row("Notebook", f"Copied -> {dst_nb.name}")

    username = _get_kaggle_username()
    if not username:
        print("  [ERROR] Cannot determine Kaggle username.")
        return 1
    kernel_id = f"{username}/{KERNEL_SLUG}"

    metadata = {
        "id": kernel_id,
        "title": "R5.5 Classifier Collapse Diagnostic",
        "code_file": NOTEBOOK_SRC.name,
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
    for fname in [NOTEBOOK_SRC.name, "kernel-metadata.json"]:
        fp = DEPLOY_DIR / fname
        passed = fp.exists()
        _print_row(fname, f"{'PASS' if passed else 'FAIL'}  ({fp.stat().st_size:,} bytes)"
                   if passed else "FAIL (missing)")
    print("\n  DEPLOYMENT READY")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    _print_section("PUSHING TO KAGGLE")
    if not DEPLOY_DIR.exists():
        print("  [ERROR] Deployment directory not found. Run 'prepare' first.")
        return 1
    kernel_id = _get_kernel_id()
    if not kernel_id:
        print("  [ERROR] Cannot determine kernel ID. Run 'prepare' first.")
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


def cmd_check(args: argparse.Namespace) -> int:
    _print_section("CHECK")
    checks = [
        ("Diagnostic notebook", NOTEBOOK_SRC),
        ("Stam config", REPO_ROOT / "training" / "config" / "stam.yaml"),
        ("Preprocessing config", REPO_ROOT / "training" / "config" / "preprocessing.yaml"),
        ("Training config", REPO_ROOT / "training" / "config" / "training.yaml"),
        ("Model config", REPO_ROOT / "training" / "config" / "model.yaml"),
        ("Frozen manifest", REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"),
        ("Frozen CSV", REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"),
    ]
    all_ok = True
    for label, path in checks:
        ok = path.exists()
        all_ok = all_ok and ok
        _print_row(label, f"{'PASS' if ok else 'FAIL'}  ({path.name})")
    if not all_ok:
        print("\n  [ERROR] One or more required files are missing.")
        return 1
    print("\n  ALL CHECKS PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-r5-5-diagnostic",
        description="Prepare/push/monitor the R5.5 diagnostic Kaggle kernel",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Copy notebook + metadata into the deploy dir")
    sub.add_parser("check", help="Verify all required source files exist")
    p_push = sub.add_parser("push", help="Push/update the notebook on Kaggle")
    p_push.add_argument(
        "--timeout", type=int, default=None, help="Optional timeout in seconds"
    )
    sub.add_parser("run", help="Push and start remote execution")
    sub.add_parser("status", help="Query kernel execution status")
    sub.add_parser("output", help="Download kernel output artifacts")
    args = parser.parse_args(argv)

    cmd = {
        "prepare": cmd_prepare,
        "check": cmd_check,
        "push": cmd_push,
        "run": cmd_run,
        "status": cmd_status,
        "output": cmd_output,
    }[args.command]
    raise SystemExit(cmd(args))


if __name__ == "__main__":  # pragma: no cover
    main()