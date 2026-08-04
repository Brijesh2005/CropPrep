#!/bin/sh
# CropFusion - model / config / report backup.
#
# Archives the filesystem model registry, generated reports, and environment
# configuration into a timestamped tarball. Run on a schedule (e.g. nightly via
# cron) or manually. Optionally uploads to an S3-compatible endpoint.
set -eu

ROOT="${ROOT:-$(pwd)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BACKUP_DIR:-${ROOT}/backups}"
TARBALL="${OUT_DIR}/cropfusion_assets_${STAMP}.tar.gz"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-14}"

mkdir -p "${OUT_DIR}"

tar -czf "${TARBALL}" \
  --exclude='**/__pycache__' \
  -C "${ROOT}" \
  models/registry reports experiments .env 2>/dev/null || true

echo "[backup] assets archived to ${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"

find "${OUT_DIR}" -name 'cropfusion_assets_*.tar.gz' -mtime +"${RETENTION_DAYS}" -delete

if [ -n "${S3_BUCKET:-}" ] && [ -n "${S3_ENDPOINT:-}" ]; then
  aws --endpoint-url "${S3_ENDPOINT}" s3 cp "${TARBALL}" "s3://${S3_BUCKET}/cropfusion/" >/dev/null
  echo "[backup] uploaded ${TARBALL}"
fi

echo "[backup] done"
