#!/usr/bin/env bash
# Launch a MATEY container for training or inference on AMD GPUs.
#
# Usage
# -----
#   ./docker_run.sh train      # run training with run_train.sh
#   ./docker_run.sh inference  # run inference with run_inference.py
#   ./docker_run.sh shell      # interactive bash shell
#   ./docker_run.sh stop       # stop and remove the container
#
# AMD Container Toolkit vs manual device flags
# --------------------------------------------
# Auto-detects AMD Container Toolkit. If not installed, falls back to
# passing /dev/kfd and /dev/dri directly.

set -euo pipefail

ROCM_IMAGE="${ROCM_IMAGE:-rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0}"
CONTAINER_NAME="${CONTAINER_NAME:-matey}"
MATEY_REPO="${MATEY_REPO:-https://github.com/ORNL/MATEY.git}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="${SCRIPT_DIR}/MATEY"

usage() {
    cat <<EOF
Usage: $(basename "$0") <mode> [args ...]

Modes:
  train      Run run_train.sh  (set MATEY_CONFIG, MATEY_RUN_NAME, MATEY_DATA_DIR)
  inference  Run run_inference.py  (set MATEY_CHECKPOINT, MATEY_INPUT)
  shell      Open an interactive bash shell inside the container
  stop       Stop and remove the running container
EOF
    exit 1
}

detect_runtime_flags() {
    if docker info 2>/dev/null | grep -qi "amd"; then
        echo "--runtime=amd -e AMD_VISIBLE_DEVICES=all"
    else
        local render_devs
        render_devs=$(ls /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')
        local dev_flags="--device=/dev/kfd"
        for dev in ${render_devs}; do
            dev_flags+=" --device=${dev}"
        done
        echo "${dev_flags} --group-add video"
    fi
}

ensure_repo() {
    if [[ ! -d "${REPO_DIR}" ]]; then
        echo "Cloning MATEY repository …"
        git clone "${MATEY_REPO}" "${REPO_DIR}"
    fi
}

ensure_container_running() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container '${CONTAINER_NAME}' exists but is stopped — removing and restarting."
            docker rm "${CONTAINER_NAME}"
        else
            echo "Container '${CONTAINER_NAME}' already running — reusing it."
            return 0
        fi
    fi

    echo "Starting container '${CONTAINER_NAME}' …"
    # shellcheck disable=SC2046
    docker run -d \
        $(detect_runtime_flags) \
        --name "${CONTAINER_NAME}" \
        --shm-size=16g \
        -v "${SCRIPT_DIR}:/workspace" \
        -v "${REPO_DIR}:/matey" \
        "${ROCM_IMAGE}" \
        tail -f /dev/null

    echo "Installing MATEY dependencies …"
    docker exec "${CONTAINER_NAME}" bash -c "
        pip install --quiet -e /matey
    "
}

ensure_repo

MODE="${1:-}"
[[ -z "${MODE}" ]] && usage
shift

case "${MODE}" in
    train)
        ensure_container_running
        echo "=== MATEY training ==="
        docker exec -it "${CONTAINER_NAME}" \
            bash /workspace/run_train.sh "$@"
        ;;
    inference)
        ensure_container_running
        echo "=== MATEY inference ==="
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
