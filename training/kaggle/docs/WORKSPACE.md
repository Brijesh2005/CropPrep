# Workspace Guide

The Kaggle workspace is the runtime directory tree the WorkspaceManager creates
and owns. It is defined by `training/config/paths.yaml` (`workspace:` section)
and resolved against the repository root.

## Layout

| Directory | Purpose | Managed by |
| --- | --- | --- |
| `training/kaggle/` | workspace root (repo-mounted) | WorkspaceManager |
| `training/kaggle/logs/` | Training Logger files (`startup.log`, `system.log`, `experiment.log`, `training.log`) | Training Logger |
| `training/kaggle/outputs/` | reports + per-run outputs | WorkspaceManager |
| `training/kaggle/outputs/reports/` | environment/gpu/dependency/storage/workspace/configuration/validation/orchestration JSON | Reports |
| `training/kaggle/checkpoints/` | checkpoint metadata registry (`metadata.json`) — no weights | CheckpointManager |
| `training/kaggle/cache/` | training cache buckets | TrainingCache |
| `training/kaggle/configs/` | resolved-config snapshot area | WorkspaceManager |

## WorkspaceManager API

`training/kaggle/workspace.py`:

- `create()` — mkdir every directory + checkpoint metadata + cache layout.
- `ensure()` — True when all directories exist and are writable.
- `clean_cache(older_than_days=None)` — clear cache entries + files.
- `output_path(*parts)` / `run_output(run_name)` / `temp_dir()`.
- `resolve_resume(run_name=None)` — delegates to the Checkpoint Manager.
- `report()` — layout + cache stats + checkpoint summary.

## Checkpoint Manager

`training/kaggle/checkpoints.py` tracks **metadata only** (no model saving):

- `list(run_name=None)` — all entries, newest first.
- `latest(run_name=None)` — most recent entry.
- `best(metric="val_loss", mode="min")` — best metric entry.
- `resume(run_name=None)` — entry flagged `resume`, else latest.
- `register(run_name, stage, epoch, metrics, path, resume)` — append + prune to
  `keep_last`.
- `version_for(run_name)` — next semver tag (`1.0.0` first, then bump patch).

## Training Cache

`training/kaggle/cache.py` — JSON-backed buckets: `metadata`,
`preprocessing`, `image_metadata`, `statistics`, `validation`. Each entry has an
optional TTL; buckets cap at `max_entries` with LRU eviction.
`get/set/has/delete/clear/section`, `stats()`.

## Reports

The Reports module (`training/kaggle/reports.py`) produces
`environment`, `gpu`, `dependency`, `storage`, `workspace`, `configuration`
reports and `write_reports()` persists them as JSON.
