#!/usr/bin/env bash
#
# Stage a self-owned copy of the VictoriaMetrics binary.
#
# Copies the portable, statically linked Go binary out of a colleague's
# read-only tree into spannala-owned space so nothing here depends on
# /shared/omnihub remaining available. Idempotent: safe to re-run.
#
set -euo pipefail

SRC="/shared/omnihub/tools/victoriametrics/victoria-metrics-prod"
DEST_DIR="/shared/spannala/perf-tools/victoriametrics"
DEST="${DEST_DIR}/victoria-metrics-prod"

if [[ ! -r "${SRC}" ]]; then
    echo "ERROR: source binary not found or unreadable: ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
cp -f "${SRC}" "${DEST}"
chmod +x "${DEST}"

echo "Staged: ${DEST}"
"${DEST}" --version
