#!/usr/bin/env bash
# Launch a StormCast container on a local workstation or interactive Docker node.
#
# Pulls the ROCm PyTorch image, installs dependencies, then runs either
# deterministic inference or ensemble forecasting.
#
# Usage
# -----
#   # Deterministic inference — 6 steps from 2025-01-01 06Z
#   ./docker_run.sh inference --start 2025-01-01T06 --steps 6
#
#   # Ensemble — 4 members, 12 steps
#   ./docker_run.sh ensemble --start 2025-08-09T12 --steps 12 --members 4
#
#   # Drop into an interactive shell inside the container
#   ./docker_run.sh shell
#
# AMD Container Toolkit vs manual device flags
# --------------------------------------------
# The script auto-detects the AMD Container Toolkit.  If it is not installed,
# it falls back to passing /dev/kfd and /dev/dri directly.
#
# See: ../recipes/inference/README.md  and  ../recipes/ensemble/README.md

set -euo pipefail

ROCM_IMAGE="${ROCM_IMAGE:-rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0}"
CONTAINER_NAME="${CONTAINER_NAME:-stormcast}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

usage() {
    cat <<EOF
Usage: $(basename "$0") <mode> [script-args ...]

Modes:
  inference   Run run_inference.py  (pass --start, --steps, --output)
  ensemble    Run run_ensemble.py   (pass --start, --steps, --members, --output)
  shell       Open an interactive bash shell inside the container
  stop        Stop and remove the running container

Examples:
  $0 inference --start 2025-01-01T06 --steps 6
  $0 ensemble  --start 2025-08-09T12 --steps 12 --members 4
  $0 shell
  $0 stop
EOF
    exit 1
}

detect_runtime_flags() {
    # Prefer AMD Container Toolkit if available; fall back to raw device flags.
    if docker info 2>/dev/null | grep -q "amd"; then
        echo "--runtime=amd -e AMD_VISIBLE_DEVICES=all"
    else
        echo "--device=/dev/kfd --device=/dev/dri --group-add video"
    fi
}

ensure_container_running() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Starting container '${CONTAINER_NAME}' …"
        # shellcheck disable=SC2046
        docker run -d \
            $(detect_runtime_flags) \
            --name "${CONTAINER_NAME}" \
            -v "${SCRIPT_DIR}:/workspace" \
            "${ROCM_IMAGE}" \
            tail -f /dev/null

        echo "Installing dependencies …"
        docker exec "${CONTAINER_NAME}" \
            pip install --quiet "earth2studio[stormcast]" cartopy
    else
        echo "Container '${CONTAINER_NAME}' already running — reusing it."
    fi
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

MODE="${1:-}"
[[ -z "${MODE}" ]] && usage
shift

case "${MODE}" in
    inference)
        ensure_container_running
        echo "=== StormCast deterministic inference ==="
        docker exec -it "${CONTAINER_NAME}" \
            python /workspace/run_inference.py "$@"
        ;;
    ensemble)
        ensure_container_running
        echo "=== StormCast ensemble inference ==="
        docker exec -it "${CONTAINER_NAME}" \
            python /workspace/run_ensemble.py "$@"
        ;;
    shell)
        ensure_container_running
        echo "Dropping into container shell — type 'exit' to leave."
        docker exec -it "${CONTAINER_NAME}" /bin/bash
        ;;
    stop)
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            docker stop "${CONTAINER_NAME}" && docker rm "${CONTAINER_NAME}"
            echo "Container '${CONTAINER_NAME}' stopped and removed."
        else
            echo "Container '${CONTAINER_NAME}' is not running."
        fi
        ;;
    *)
        echo "error: unknown mode '${MODE}'" >&2
        usage
        ;;
esac
