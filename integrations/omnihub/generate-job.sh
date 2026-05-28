#!/usr/bin/env bash
# Generate an OmniHub SLURM job for ai4science HydraGNN with Lux-specific apptainer binds.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AI4S_REPO_ROOT="${AI4S_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OMNIHUB_DIR="${OMNIHUB_DIR:-}"
AI4S_SHARED_DIR="${AI4S_SHARED_DIR:-}"

NUM_NODES=1
PARTITION=""
CLUSTER=vultr
TIME_LIMIT=2h
TOOLS=(omnihub-monitor)
RUNNER=manual-mpi
TASKS_PER_NODE=8
OUTPUT=""
IMAGE=""
APP_CONFIG=applications/ai4science-hydragnn-train/config.yaml
ROCM_VERSION=7.2.2
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options]

  --omnihub-dir PATH     OmniHub repo (default: OMNIHUB_DIR env)
  --num-nodes N          SLURM nodes (default: 1)
  --partition NAME       SLURM partition (default: from .cluster-config.yaml)
  --cluster NAME         OmniHub cluster yaml (default: vultr)
  --time-limit LIMIT     e.g. 2h, 30m
  --tools NAME           Repeatable; default: omnihub-monitor
  --runner NAME          manual-mpi | manual | torchrun (default: manual-mpi)
  --tasks-per-node N     Default: 8 for HydraGNN
  --image PATH           Apptainer SIF (default: AI4S pytorch rocm7.2.2)
  --output FILE          Write SLURM script here (default: stdout)
  --app-config PATH      Relative to omnihub dir
  --perf                 Shortcut: omnihub-monitor omnistat pytorch-trace tracelens
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --omnihub-dir) OMNIHUB_DIR=$2; shift 2 ;;
    --num-nodes) NUM_NODES=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --cluster) CLUSTER=$2; shift 2 ;;
    --time-limit) TIME_LIMIT=$2; shift 2 ;;
    --tools) TOOLS+=("$2"); shift 2 ;;
    --runner) RUNNER=$2; shift 2 ;;
    --tasks-per-node) TASKS_PER_NODE=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --app-config) APP_CONFIG=$2; shift 2 ;;
    --perf) TOOLS=(omnihub-monitor omnistat pytorch-trace tracelens); APP_CONFIG=applications/ai4science-hydragnn-train/config-perf.yaml; shift ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

: "${OMNIHUB_DIR:?Set OMNIHUB_DIR or pass --omnihub-dir}"
# Compute nodes may not see $HOME; prefer a shared checkout when present.
OMNIHUB_CLUSTER_DIR="${OMNIHUB_CLUSTER_DIR:-$AI4S_SHARED_DIR/omnihub}"
if [[ -x "$OMNIHUB_CLUSTER_DIR/omnihub-generate-job" ]]; then
  OMNIHUB_DIR="$OMNIHUB_CLUSTER_DIR"
fi
: "${AI4S_SHARED_DIR:?Set AI4S_SHARED_DIR}"

"$SCRIPT_DIR/sync-to-omnihub.sh"

HG_BASE="$AI4S_SHARED_DIR/models/HydraGNN"
HG_OVERLAY="${HG_OVERLAY:-$HG_BASE/overlays/hydragnn-overlay-pre-mainbump.img}"
if [[ ! -f "$HG_OVERLAY" ]]; then
  HG_OVERLAY="$HG_BASE/overlays/hydragnn-overlay.img"
fi
HG_SIF="${HG_SIF:-$AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
HG_REPO="${HG_REPO_DIR:-$HG_BASE/code/HydraGNN}"
HG_DATA="${HG_DATA_DIR:-$HG_BASE/weights}"
SCRATCH="${SCRATCH_LOCAL:-/scratch}"

if [[ -z "$PARTITION" && -f "$AI4S_REPO_ROOT/.cluster-config.yaml" ]]; then
  PARTITION=$(grep -E '^[[:space:]]*partition:' "$AI4S_REPO_ROOT/.cluster-config.yaml" | head -1 | sed -E 's/.*:[[:space:]]*"?([^"]*)"?/\1/')
fi
: "${PARTITION:?Set --partition or slurm.partition in .cluster-config.yaml}"

if [[ -z "$IMAGE" ]]; then
  IMAGE="$HG_SIF"
fi

TOOL_FLAGS=(--tools "${TOOLS[@]}")

GEN_ARGS=(
  --omnihub-dir "$OMNIHUB_DIR"
  --app-config "$APP_CONFIG"
  --cluster "$CLUSTER"
  --partition "$PARTITION"
  --num-nodes "$NUM_NODES"
  --runner "$RUNNER"
  --tasks-per-node "$TASKS_PER_NODE"
  --time-limit "$TIME_LIMIT"
  --rocm-version "$ROCM_VERSION"
  --image "$IMAGE"
  "${TOOL_FLAGS[@]}"
)

if [[ -n "$OUTPUT" ]]; then
  GEN_ARGS+=(--output "$OUTPUT")
fi

# Export HydraGNN paths for job template / wrapper
export AI4S_SHARED_DIR HG_OVERLAY HG_REPO_DIR="$HG_REPO" HG_DATA_DIR="$HG_DATA"
export HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-$AI4S_SHARED_DIR/results/omnihub}"
export SCRATCH_LOCAL="$SCRATCH"

# Patch apptainer exec line to add rocm + HydraGNN ext3 overlay
overlay_bind=""
if [[ -f "$HG_OVERLAY" ]]; then
  overlay_bind="--overlay ${HG_OVERLAY}:ro"
fi
export OMNIHUB_APPTAINER_EXEC_EXTRA=""
export OMNIHUB_APPTAINER_BINDS="/opt/ompi:/opt/ompi:ro,${SCRATCH}:${SCRATCH},${HG_REPO}:${HG_REPO}:ro,${HG_DATA}:${HG_DATA}:ro,${AI4S_SHARED_DIR}:${AI4S_SHARED_DIR},${OMNIHUB_DIR}:${OMNIHUB_DIR}:ro"

TMP_SLURM=$(mktemp)
"$OMNIHUB_DIR/omnihub-generate-job" "${GEN_ARGS[@]}" --output "$TMP_SLURM"

# Inject HydraGNN env + apptainer patches before sanity-check block
python3 - "$TMP_SLURM" <<'PY'
import os, sys
path = sys.argv[1]
overlay = os.environ.get("HG_OVERLAY", "")
extra_binds = os.environ.get("OMNIHUB_APPTAINER_BINDS", "")
bind_flags = " ".join(f"--bind {b}" for b in extra_binds.split(",") if b.strip())
text = open(path).read()
inject = (
    '# AI4Science Studio — HydraGNN paths\n'
    f'export AI4S_SHARED_DIR="{os.environ.get("AI4S_SHARED_DIR", "")}"\n'
    f'export HG_REPO_DIR="{os.environ.get("HG_REPO_DIR", "")}"\n'
    f'export HG_DATA_DIR="{os.environ.get("HG_DATA_DIR", "")}"\n'
    f'export HG_OVERLAY="{overlay}"\n'
    'export PYTHONPATH="/opt/hydragnn-pkgs:${PYTHONPATH:-}"\n'
    'export LD_LIBRARY_PATH="/opt/hydragnn-pkgs/adios2:/opt/ompi/lib:${LD_LIBRARY_PATH:-}"\n'
    f'export SCRATCH_LOCAL="{os.environ.get("SCRATCH_LOCAL", "/scratch")}"\n'
    'export OMNIHUB_SANITY_APPTAINER_FLAGS="--rocm"\n'
    'export OMNIHUB_SANITY_GRES=""\n'
    'export OMNIHUB_SKIP_NCCL_SANITY="1"\n'
    f'apptainer_bind_args="$apptainer_bind_args {bind_flags}"\n'
)
marker = "# Sanity checks:"
if marker not in text:
    sys.exit("Could not find sanity-check marker in generated SLURM script")
text = text.replace(marker, inject + marker, 1)
# Lux MI355X: request GPUs explicitly (omnihub job template omits this)
if "#SBATCH --gpus-per-node" not in text:
    text = text.replace(
        "#SBATCH --tasks-per-node=",
        f"#SBATCH --gpus-per-node={os.environ.get('OMNIHUB_GPUS_PER_NODE', '8')}\n#SBATCH --tasks-per-node=",
        1,
    )
if overlay:
    # HydraGNN uses a read-only ext3 overlay; stacking it with OmniHub's per-task
    # writable overlay triggers fuse2fs timeouts on Lux. Use the HydraGNN overlay
    # alone (same pattern as sbatch_train_amd.sh).
    text = text.replace(
        "mkdir -p /tmp/omnihub-overlay.\\$SLURM_PROCID/{upper,work} && ",
        "",
    )
    text = text.replace(" --overlay /tmp/omnihub-overlay.\\$SLURM_PROCID", "")
    if f"--overlay {overlay}:ro" not in text:
        text = text.replace(
            "apptainer exec --rocm ",
            f"apptainer exec --rocm --overlay {overlay}:ro ",
            1,
        )
    text = text.replace("--rocm --rocm --overlay", "--rocm --overlay")
# Ensure HydraGNN pkg paths survive omnihub-apptainer-env.sh inside the container
container_env = (
    'export PYTHONPATH="/opt/hydragnn-pkgs:${PYTHONPATH:-}"; '
    'export LD_LIBRARY_PATH="/opt/hydragnn-pkgs/adios2:/opt/ompi/lib:${LD_LIBRARY_PATH:-}"; '
)
needle = "source $omnihub_dir/scripts/omnihub-apptainer-env.sh; "
if needle in text:
    text = text.replace(needle, needle + container_env, 1)
open(path, "w").write(text)
PY

if [[ -n "$OUTPUT" ]]; then
  mv "$TMP_SLURM" "$OUTPUT"
  chmod +x "$OUTPUT"
  echo "Wrote $OUTPUT"
else
  cat "$TMP_SLURM"
  rm -f "$TMP_SLURM"
fi
