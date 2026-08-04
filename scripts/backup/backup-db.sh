#!/bin/sh
# CropFusion - PostgreSQL backup (runs inside the `backup` compose service).
#
# dumps POSTGRES_DB on POSTGRES_HOST daily, compresses it, prunes old dumps,
# and (optionally) copies the archive to an S3-compatible endpoint.
set -eu

BACKUP_DIR="${BACKUP_DIR:-/backups}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DUMP="${BACKUP_DIR}/cropfusion_${STAMP}.sql.gz"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-14}"

echo "[backup] starting dump of ${POSTGRES_DB:-cropfusion}"

apk add --no-cache postgresql16-client >/dev/null 2>&1 || apk add --no-cache postgresql-client >/dev/null

mkdir -p "${BACKUP_DIR}"
pg_dump -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER:-cropfusion}" \
  -d "${POSTGRES_DB:-cropfusion}" --no-owner --no-privileges \
  | gzip -9 > "${DUMP}"

echo "[backup] written ${DUMP} ($(du -h "${DUMP}" | cut -f1))"

# Prune local retention.
find "${BACKUP_DIR}" -name 'cropfusion_*.sql.gz' -mtime +"${RETENTION_DAYS}" -delete

# Optional offsite copy.
if [ -n "${S3_BUCKET:-}" ] && [ -n "${S3_ENDPOINT:-}" ]; then
  apk add --no-cache aws-cli >/dev/null 2>&1 || true
  aws --endpoint-url "${S3_ENDPOINT}" s3 cp "${DUMP}" "s3://${S3_BUCKET}/cropfusion/" \
    >/dev/null
  echo "[backup] uploaded ${DUMP} to s3://${S3_BUCKET}/cropfusion/"
fi

echo "[backup] done"
