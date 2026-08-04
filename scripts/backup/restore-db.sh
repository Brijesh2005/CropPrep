#!/bin/sh
# CropFusion - restore a PostgreSQL dump.
#
# Usage:
#   POSTGRES_HOST=localhost POSTGRES_USER=cropfusion POSTGRES_DB=cropfusion \
#   RESTORE_FILE=/backups/cropfusion_20260101_000000.sql.gz ./restore-db.sh
set -eu

RESTORE_FILE="${RESTORE_FILE:-}"
if [ -z "${RESTORE_FILE}" ]; then
  echo "error: RESTORE_FILE must point to a .sql.gz dump" >&2
  exit 1
fi

echo "[restore] restoring ${RESTORE_FILE} -> ${POSTGRES_DB:-cropfusion}"

gunzip -c "${RESTORE_FILE}" \
  | psql -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER:-cropfusion}" \
         -d "${POSTGRES_DB:-cropfusion}" --set ON_ERROR_STOP=1

echo "[restore] done"
