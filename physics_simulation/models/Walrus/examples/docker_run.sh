#!/usr/bin/env bash
# Launch a Walrus container for inference on AMD GPUs.
#
# Usage
# -----
#   ./docker_run.sh inference  # run inference with run_inference.py
#   ./docker_run.sh shell      # interactive bash shell
#   ./docker_run.sh stop       # stop and remove the container

set -euo pipefail

ROCM_IMAGE="${ROCM_IMAGE:-rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1}"
CONTAINER_NAME="${CONTAINER_NAME:-walrus}"
WALRUS_REPO="${WALRUS_REPO:-https://github.com/PolymathicAI/walrus.git}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="${SCRIPT_DIR}/walrus"

usage() {
    cat <<EOF
Usage: $(basename "$0") <mode> [args ...]

Modes:
  inference  Run run_inference.py (set WALRUS_WEIGHTS_DIR, WALRUS_INPUT)
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
        echo "Cloning Walrus repository …"
        git clone "${WALRUS_REPO}" "${REPO_DIR}"
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
        -v "${REPO_DIR}:/walrus" \
        "${ROCM_IMAGE}" \
        tail -f /dev/null

    echo "Installing Walrus dependencies …"
    docker exec "${CONTAINER_NAME}" bash -c "
        pip install --quiet huggingface-hub
        pip install --quiet -e /walrus
    "
}

ensure_repo

MODE="${1:-}"
[[ -z "${MODE}" ]] && usage
shift

case "${MODE}" in
    inference)
        ensure_container_running
        echo "=== Walrus inference ==="
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
