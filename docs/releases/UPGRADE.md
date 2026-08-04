# Upgrade Guide

How to upgrade a CropFusion deployment between versions.

## Before upgrading

1. **Back up** the database and assets:
   ```bash
   docker compose -f docker-compose.prod.yml run --rm backup
   # plus an assets backup via scripts/backup/backup-assets.sh
   ```
2. **Read the release notes** (`docs/releases/`) for the target version.
3. On a staging copy, apply the upgrade first if possible.

## Steps

```bash
cd /srv/cropfusion

# 1. Pull the new images
docker compose -f docker-compose.prod.yml pull

# 2. Apply database migrations
docker compose -f docker-compose.prod.yml run --rm backend \
  alembic upgrade head

# 3. Recreate services
docker compose -f docker-compose.prod.yml up -d
```

## Smoke test

- `curl -fsS https://<domain>/health`
- Open the frontend and run a prediction.
- Check Grafana for backend uptime and error-rate alerts.

## Rollback

- **Application:** redeploy the previous image tags
  (`docker compose ... up -d` with the prior tag).
- **Database:** restore from `scripts/backup/restore-db.sh` if a migration
  must be undone (never mix partial migrations across versions).
- **Model:** `cropfusion-mlops rollback <model> <version>`.

## Version history

| Version | Notes |
|---|---|
| 1.0.0 | First stable release. |
