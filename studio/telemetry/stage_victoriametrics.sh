#!/usr/bin/env bash
#
# Stage a self-owned copy of the VictoriaMetrics binary.
#
# Copies the portable, statically linked Go binary from an existing install
# (or a release download from github.com/VictoriaMetrics/VictoriaMetrics) into
# $PERF_TOOLS_DIR, so nothing here depends on someone else's tree remaining
# available. Idempotent: safe to re-run.
#
# Env:
#   VM_BINARY_SRC   path to an existing victoria-metrics-prod binary (required)
#   PERF_TOOLS_DIR  destination root (default: $AI4S_SHARED_DIR/perf-tools)
#
set -euo pipefail

SRC="${VM_BINARY_SRC:?set VM_BINARY_SRC to a victoria-metrics-prod binary}"
PERF_TOOLS_DIR="${PERF_TOOLS_DIR:-${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR or PERF_TOOLS_DIR}/perf-tools}"
DEST_DIR="${PERF_TOOLS_DIR}/victoriametrics"
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
