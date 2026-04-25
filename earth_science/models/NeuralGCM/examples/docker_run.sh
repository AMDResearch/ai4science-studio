#!/usr/bin/env bash
# Launch a NeuralGCM container on a local workstation or interactive Docker node.
#
# Pulls the ROCm dev image, installs JAX for ROCm and NeuralGCM, then runs
# inference using run_inference.py.
#
# Usage
# -----
#   # 4-day deterministic forecast at 1.4° from 2020-01-01
#   ./docker_run.sh inference --date 2020-01-01 --steps 16
#
#   # 10-day stochastic forecast at 1.4°
#   ./docker_run.sh inference \
#       --checkpoint v1/stochastic_1_4_deg.pkl --steps 40 --seed 42
#
#   # Drop into an interactive shell inside the container
#   ./docker_run.sh shell
#
#   # Stop and remove the container
#   ./docker_run.sh stop
#
# AMD Container Toolkit vs manual device flags
# --------------------------------------------
# The script auto-detects the AMD Container Toolkit.  If it is not installed,
# it falls back to passing /dev/kfd and /dev/dri directly.

set -euo pipefail

ROCM_IMAGE="${ROCM_IMAGE:-rocm/dev-ubuntu-22.04:7.0.2-complete}"
CONTAINER_NAME="${CONTAINER_NAME:-neuralgcm}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# JAX ROCm wheel
JAX_WHEEL="https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.0-cp310-cp310-manylinux2014_x86_64.whl"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

usage() {
    cat <<EOF
Usage: $(basename "$0") <mode> [script-args ...]

Modes:
  inference   Run run_inference.py  (pass --checkpoint, --date, --steps, --output)
  shell       Open an interactive bash shell inside the container
  stop        Stop and remove the running container

Examples:
  $0 inference --date 2020-01-01 --steps 16
  $0 inference --checkpoint v1/stochastic_1_4_deg.pkl --steps 40 --seed 42
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

        echo "Installing JAX for ROCm and NeuralGCM …"
        docker exec "${CONTAINER_NAME}" bash -c "
            python3 -m pip install --quiet ${JAX_WHEEL}
            python3 -m pip install --quiet jax==0.6.0 jax-rocm7-pjrt jax-rocm7-plugin
            apt-get update -qq && apt-get install -y -qq libdw1
            python3 -m pip install --quiet neuralgcm gcsfs xarray matplotlib
        "
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
        echo "=== NeuralGCM inference ==="
        docker exec -it "${CONTAINER_NAME}" \
            python3 /workspace/run_inference.py "$@"
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
