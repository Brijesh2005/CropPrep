"""CropFusion MLOps CLI.

Usage::

    cropfusion-mlops register yieldnet 1.2.0 model.pt --accuracy 0.87 --loss 0.31
    cropfusion-mlops gate yieldnet 1.2.0 --accuracy 0.87
    cropfusion-mlops promote yieldnet 1.2.0 --target production
    cropfusion-mlops rollback yieldnet 1.0.0
    cropfusion-mlops list --status production
    cropfusion-mlops experiment log --model yieldnet --accuracy 0.87
    cropfusion-mlops benchmark run --model yieldnet --version 1.2.0
    cropfusion-mlops scheduler --once
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from .config import load_settings
from .experiments import ExperimentTracker
from .gates import GateResult, all_passed, metric_gate, write_gate_report
from .registry import ModelRegistry
from .reports import write_release_report

app = typer.Typer(help="CropFusion MLOps toolkit", no_args_is_help=True)


def _registry() -> tuple[ModelRegistry, Any]:
    settings = load_settings()
    return ModelRegistry(settings), settings


@app.command()
def register(
    name: str,
    version: str,
    checkpoint: str,
    accuracy: float = typer.Option(0.0, help="validation accuracy (0-1)"),
    loss: float = typer.Option(0.0, help="validation loss"),
    git_commit: str = typer.Option(None, help="source commit"),
    notes: str = typer.Option(None),
) -> None:
    """Register a trained checkpoint as a draft model version."""
    registry, _ = _registry()
    record = registry.register(
        name,
        version,
        checkpoint_path=checkpoint,
        metrics={"accuracy": accuracy, "loss": loss} if accuracy or loss else None,
        git_commit=git_commit,
        notes=notes,
    )
    typer.echo(f"registered {name}@{version} (id={record.manifest.id})")


@app.command()
def gate(
    name: str,
    version: str,
    accuracy: float = typer.Option(0.0),
    incumbent: str = typer.Option(None, help="incumbent version to compare against"),
) -> None:
    """Run validation gates without promoting."""
    registry, settings = _registry()
    record = registry.get(name, version)
    gates: list[GateResult] = []
    if accuracy:
        incumbent_metrics = None
        if incumbent:
            incumbent_metrics = registry.get(name, incumbent).manifest.metrics
        gates.append(metric_gate(record.manifest.metrics or {"accuracy": accuracy}, settings, incumbent_metrics=incumbent_metrics))
    write_gate_report(gates, settings.reports_dir)
    for g in gates:
        typer.echo(f"[{'PASS' if g.passed else 'FAIL'}] {g.gate}: {g.message}")
    if not all_passed(gates):
        raise typer.Exit(1)


@app.command()
def promote(
    name: str,
    version: str,
    target: str = typer.Option("production", help="staging | production"),
    accuracy: float = typer.Option(0.0, help="gate: validation accuracy"),
    promoted_by: str = typer.Option(None),
) -> None:
    """Promote a model version to staging or production."""
    registry, settings = _registry()
    gates: list[GateResult] = []
    if accuracy:
        incumbent = registry.active(name)
        gates.append(
            metric_gate(
                registry.get(name, version).manifest.metrics or {"accuracy": accuracy},
                settings,
                incumbent_metrics=incumbent.manifest.metrics if incumbent else None,
            )
        )
    if not all_passed(gates):
        typer.secho("promotion blocked by failing gates", fg="red")
        raise typer.Exit(1)
    record = registry.promote(name, version, target=target, gates=gates, promoted_by=promoted_by)
    write_release_report(settings, model_name=name, version=version, target=target, gates=gates, registry=registry)
    typer.secho(f"{name}@{version} promoted to {target}", fg="green")


@app.command()
def rollback(
    name: str,
    version: str,
    promoted_by: str = typer.Option(None),
) -> None:
    """Roll the active production model back to a previous version."""
    registry, _ = _registry()
    registry.rollback(name, version, promoted_by=promoted_by)
    typer.secho(f"rolled back {name} to {version}", fg="green")


@app.command("list")
def list_models(
    name: str = typer.Option(None),
    status: str = typer.Option(None, help="draft|staging|production|archived"),
) -> None:
    """List registered model versions."""
    registry, _ = _registry()
    for record in registry.list(name=name, status=status):
        m = record.manifest
        typer.echo(f"{m.name}@{m.version} [{m.status}] acc={m.metrics.get('accuracy', 'n/a')}")


@app.command()
def info(name: str, version: str) -> None:
    """Show a model version manifest."""
    registry, _ = _registry()
    typer.echo(registry.get(name, version).manifest.model_dump_json(indent=2))


@app.command()
def archive(name: str, version: str, notes: str = typer.Option(None)) -> None:
    """Archive a non-production model version."""
    registry, _ = _registry()
    registry.archive(name, version, notes=notes)
    typer.secho(f"archived {name}@{version}", fg="green")


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #


@app.command()
def experiment(
    model: str,
    accuracy: float = typer.Option(0.0),
    loss: float = typer.Option(0.0),
    dataset_version: str = typer.Option(None),
    notes: str = typer.Option(None),
) -> None:
    """Record one training run in the experiment log."""
    _, settings = _registry()
    tracker = ExperimentTracker(settings)
    run = tracker.log(
        model_name=model,
        config={},
        metrics={"accuracy": accuracy, "loss": loss},
        dataset_version=dataset_version,
        notes=notes,
    )
    typer.echo(f"recorded run {run['run_id']}")


@app.command()
def experiments_list(model: str = typer.Option(None)) -> None:
    """List recorded experiment runs."""
    _, settings = _registry()
    tracker = ExperimentTracker(settings)
    for run in tracker.runs(model):
        typer.echo(f"{run['timestamp']} {run['model_name']} {run['run_id']} acc={run['metrics'].get('accuracy')}")


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


@app.command()
def scheduler(
    once: bool = typer.Option(False, help="run a single cycle and exit"),
    interval: int = typer.Option(None),
) -> None:
    """Run the monitoring scheduler (admin container entrypoint)."""
    from .scheduler import main as scheduler_main

    argv: list[str] = []
    if once:
        argv.append("--once")
    if interval:
        argv += ["--interval", str(interval)]
    raise SystemExit(scheduler_main(argv))


if __name__ == "__main__":
    app()
