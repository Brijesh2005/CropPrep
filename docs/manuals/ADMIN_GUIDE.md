# Administrator Guide

This guide covers day-to-day operation of a CropFusion deployment: users,
monitoring, model management and maintenance. For infrastructure deployment,
see [DEPLOYMENT.md](../DEPLOYMENT.md).

## Roles

CropFusion uses role-based access control:

| Role | Capabilities |
|---|---|
| Farmer | predictions, history, own locations, notifications |
| Admin | user management, audit log, dashboard, model/dataset registry |

## Admin dashboard

Admins can open **Admin > Dashboard** to see:

- System health and request volumes.
- Prediction activity and model version in use.
- Drift/fairness status (from the MLOps scheduler).

The **Audit log** records administrative actions (registrations, promotions,
configuration changes) for compliance.

## User management

- View and search users.
- Reset passwords (generates a reset token sent to the user's email).
- Lock/unlock accounts after repeated failed logins (automatic lockout after
  `max_failed_attempts`).

## Model management

Models are managed through the MLOps CLI (see [MLOPS.md](../MLOPS.md)):

```bash
# View what is live
cropfusion-mlops list --status production

# Promote a validated model
cropfusion-mlops promote yieldnet 1.2.0 --target production --accuracy 0.87

# Roll back on issues
cropfusion-mlops rollback yieldnet 1.0.0
```

Promotion gates (accuracy, latency regression, drift, fairness) must pass
before a model can serve traffic. The registry API (`/api/v1/registry`) is the
programmatic equivalent for UI integration.

## Monitoring

- **Grafana** at `/grafana/` (prod) or `:3001` (dev) - dashboards for ML
  quality and performance.
- **Prometheus** at `/metrics` - alerts for backend down, high error rate,
  high latency, drift and low disk space (`deployment/monitoring/alerts.yml`).
- **Loki** - centralised logs; correlate with request IDs.

## Backups

- The `backup` service dumps the database daily (retention:
  `BACKUP_RETENTION_DAYS`).
- Model/registry/reports backups: `scripts/backup/backup-assets.sh`.
- Restore: `scripts/backup/restore-db.sh` (see [DEPLOYMENT.md](../DEPLOYMENT.md#backups)).

Verify backups periodically by restoring into a scratch database.

## Maintenance

- **Migrations**: run `alembic upgrade head` in the backend container after
  upgrading.
- **Upgrades**: pull new images and recreate services
  (see [DEPLOYMENT.md](../DEPLOYMENT.md#upgrades--rollback)).
- **Scheduler**: the `admin` container runs drift/fairness checks; confirm it
  is healthy and producing reports under `reports/`.
