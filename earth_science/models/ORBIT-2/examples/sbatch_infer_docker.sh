#!/usr/bin/env bash
# ORBIT-2 visualization / inference on AMD Instinct via SLURM + Docker.
#
# Unlike the Apptainer version (sbatch_infer_amd.sh) which uses
# srun --mpi=pmix with per-rank containers, this script runs a SINGLE
# Docker container with all GPUs and uses torchrun for distributed launch.
# This avoids Docker+PMIx incompatibility.
#
# Quick-start with synthetic data (no real ERA5/PRISM data required):
#   export ORBIT2_ROOT=/path/to/ORBIT-2-clone
#   export ORBIT2_USE_SYNTHETIC=1
#   sbatch sbatch_infer_docker.sh
#
# Prerequisites:
#   1. Docker must be available on compute nodes.
#   2. Clone https://github.com/XiaoWang-Github/ORBIT-2 into ORBIT2_ROOT.
#   3. Download a pretrain checkpoint from https://huggingface.co/jychoi-hpc/ORBIT-2
#      and set ORBIT2_CHECKPOINT to its path (auto-downloaded if unset + synthetic).
#   4. Internet access from compute nodes is required to pull the Docker image
#      and download HuggingFace weights on first run.
#
# GPU / ROCm compatibility:
#   Default image covers MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950).
#   For older hardware (MI100/gfx908): override ORBIT2_IMAGE with a rocm6.x image.
#
# Key environment variables:
#   ORBIT2_IMAGE          Docker image (default: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0)
#   ORBIT2_ROOT           Path to ORBIT-2 clone (required)
#   ORBIT2_CHECKPOINT     Path to .ckpt file (auto-downloaded if unset + synthetic)
#   ORBIT2_CONFIG         Config YAML basename (default: interm_8m_ft.yaml, ignored if synthetic)
#   ORBIT2_USE_SYNTHETIC  Set to 1 for synthetic smoke test (no real data needed)
#   ORBIT2_NPROC          GPUs per node (default: 8)
#   ROCM_WHL_TAG          PyTorch wheel index suffix for xformers (default: rocm7.2)
#
# See ../recipes/inference-and-visualization.md and ../recipes/local-cluster-amd.md

##SBATCH -A YOUR_PROJECT_HERE
#SBATCH -J orbit2-docker
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH -t 00:30:00
#SBATCH -o orbit2-docker-%j.out
#SBATCH -e orbit2-docker-%j.out

set -euo pipefail

# SLURM copies the script to its spool dir before execution, so BASH_SOURCE[0]
# would resolve to the spool path. Use scontrol to recover the original path.
SCRIPT_DIR=$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')")" && pwd)

if [[ -z "${ORBIT2_ROOT:-}" ]]; then
  echo "error: export ORBIT2_ROOT to your ORBIT-2 clone path" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORBIT2_IMAGE="${ORBIT2_IMAGE:-rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
ORBIT2_NPROC="${ORBIT2_NPROC:-8}"
ROCM_WHL_TAG="${ROCM_WHL_TAG:-rocm7.2}"
CONTAINER_NAME="orbit2-${SLURM_JOB_ID:-$$}"

# MIOpen cache dir (writable tmpdir mapped into container)
MIOPEN_HOST="${TMPDIR:-/tmp}/orbit2-miopen-${SLURM_JOB_ID:-$$}"
mkdir -p "$MIOPEN_HOST"

echo "=== ORBIT-2 inference (Docker) ==="
echo "  Image        : $ORBIT2_IMAGE"
echo "  ORBIT2_ROOT  : $ORBIT2_ROOT"
echo "  GPUs         : $ORBIT2_NPROC"
echo "  Container    : $CONTAINER_NAME"
echo "  Job          : ${SLURM_JOB_ID:-local}"
echo ""

# ---------------------------------------------------------------------------
# Detect GPU access method
# ---------------------------------------------------------------------------
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi "amd"; then
    GPU_ARGS=(--runtime=amd -e AMD_VISIBLE_DEVICES=all)
    echo "  GPU method : AMD Container Toolkit"
else
    GPU_ARGS=(--device=/dev/kfd)
    for dev in /dev/dri/renderD*; do
        [[ -e "$dev" ]] && GPU_ARGS+=(--device="$dev")
    done
    GPU_ARGS+=(--group-add video)
    echo "  GPU method : device passthrough"
fi
echo ""

# ---------------------------------------------------------------------------
# Synthetic data setup (host-side config preparation)
# ---------------------------------------------------------------------------
if [[ "${ORBIT2_USE_SYNTHETIC:-0}" == "1" ]]; then
  SYNTH_DIR="${TMPDIR:-/tmp}/orbit2-synthetic-${SLURM_JOB_ID:-$$}"
  mkdir -p "$SYNTH_DIR"

  # Stamp data path into a temp copy of the synthetic config (host-side).
  # Data path is /synth_data inside the container — stamp it now.
  # Checkpoint path is stamped inside the container after download.
  SYNTH_CONFIG="${TMPDIR:-/tmp}/interm_8m_synthetic_${SLURM_JOB_ID:-$$}.yaml"
  cp "$SCRIPT_DIR/interm_8m_synthetic.yaml" "$SYNTH_CONFIG"
  sed -i "s|__SYNTH_DATA_DIR__|/synth_data|g" "$SYNTH_CONFIG"

  CONFIG="$SYNTH_CONFIG"

  if [[ -n "${ORBIT2_CHECKPOINT:-}" ]]; then
    CKPT="$ORBIT2_CHECKPOINT"
  else
    CKPT=""  # will be downloaded inside container
  fi
else
  CONFIG="${ORBIT2_CONFIG:-interm_8m_ft.yaml}"
  CKPT="${ORBIT2_CHECKPOINT:-}"
  SYNTH_DIR=""
fi

# ---------------------------------------------------------------------------
# Write the setup-and-run script (runs inside the container)
# ---------------------------------------------------------------------------
RUN_SCRIPT="${TMPDIR:-/tmp}/orbit2_docker_run_${SLURM_JOB_ID:-$$}.sh"
cat > "$RUN_SCRIPT" << 'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/orbit2/src:/orbit2:${PYTHONPATH:-}"

ROCM_WHL_TAG="$1"
NPROC="$2"
USE_SYNTHETIC="$3"
CKPT="$4"
shift 4
# Remaining args passed to run_visualize.py

echo "--- Installing ORBIT-2 dependencies ---"

# Core deps
pip install -q --no-cache-dir \
    --extra-index-url "https://download.pytorch.org/whl/${ROCM_WHL_TAG}" \
    mpi4py huggingface-hub 2>&1 | tail -3

# pytorch-lightning without torch dep
pip install -q --no-cache-dir --no-deps pytorch-lightning 2>&1 | tail -3
pip install -q --no-cache-dir \
    --extra-index-url "https://download.pytorch.org/whl/${ROCM_WHL_TAG}" \
    lightning-utilities torchmetrics 2>&1 | tail -3

# Science / data deps
pip install -q --no-cache-dir \
    --extra-index-url "https://download.pytorch.org/whl/${ROCM_WHL_TAG}" \
    "timm==0.9.2" "tensorboard==2.11.2" wandb \
    cdsapi "dask>=2022.2.0" "importlib-metadata==4.13.0" \
    "matplotlib>=3.5.3" "netcdf4>=1.6.2" "scikit-learn>=1.0.2" \
    "xarray>=0.20.2" "rasterio>=1.3.7" scikit-image einops lpips \
    pyyaml 2>&1 | tail -5

# xformers for ROCm
pip install -q --no-cache-dir --no-deps -U xformers \
    --index-url "https://download.pytorch.org/whl/${ROCM_WHL_TAG}" 2>&1 | tail -3

# xformers.components shim (removed in xformers >=0.0.28)
python3 - << 'PYEOF'
import pathlib, importlib
xf = pathlib.Path(importlib.import_module("xformers").__file__).parent
comp = xf / "components"; attn = comp / "attention"
for d in [comp, attn]: d.mkdir(exist_ok=True)
(comp / "__init__.py").write_text("")
(attn / "__init__.py").write_text("")
(attn / "core.py").write_text(
    "import torch.nn.functional as F\n\n"
    "def scaled_dot_product_attention(q, k, v, att_mask=None, dropout=0.0):\n"
    "    return F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask, dropout_p=dropout)\n"
)
print("xformers.components shim installed")
PYEOF

# mpi4py stub — Docker has no OpenMPI runtime, but distdataset.py imports mpi4py
# at module level (via climate_learn → itermodule.py → distdataset.py).
# ORBIT_USE_DDSTORE=0 (default) means no actual MPI calls are made; we only need
# the import to succeed. The stub satisfies all import-time references.
mkdir -p /tmp/mpi4py_stub/mpi4py
python3 - << 'PYEOF'
stub = '''\
class _RC:
    thread_level = "serialized"
    threads = False
    initialize = False
rc = _RC()
class _CommWorld:
    rank = 0; size = 1
    def allreduce(self, x, op=None): return x
    def Barrier(self): pass
    def bcast(self, obj, root=0): return obj
class MPI:
    COMM_WORLD = _CommWorld()
    SUM = 0; MAX = 1; MIN = 2; LAND = 3; LOR = 4; IN_PLACE = None
'''
open('/tmp/mpi4py_stub/mpi4py/__init__.py', 'w').write(stub)
print("mpi4py stub installed at /tmp/mpi4py_stub/mpi4py")
PYEOF
export PYTHONPATH="/tmp/mpi4py_stub:${PYTHONPATH}"

# In Docker (unlike Apptainer SIF), the container filesystem is writable.
# pip should keep the existing ROCm torch if it satisfies version constraints.
# Do NOT strip torch here — it would remove the only copy.
# Only strip nvidia/triton CUDA co-packages which are safe to remove.
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
for pkg in nvidia triton; do
    rm -rf "${SITE}/${pkg}" "${SITE}/${pkg}"-*.dist-info 2>/dev/null || true
done

# Verify ROCm torch is intact
python3 -c "
import torch
assert 'rocm' in torch.__version__, f'Non-ROCm torch: {torch.__version__}'
print('torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())
"

# --- Synthetic data generation (Phase 1) ---
if [[ "$USE_SYNTHETIC" == "1" ]]; then
    echo "--- Generating synthetic ORBIT-2 dataset ---"
    python3 /examples/make_synthetic_data.py --out-dir /synth_data

    # Download checkpoint if not provided
    if [[ -z "$CKPT" ]]; then
        echo "--- Downloading smallest pretrain checkpoint from HF ---"
        CKPT=$(python3 - << 'PYEOF'
from huggingface_hub import list_repo_files, hf_hub_download
import os, glob

repo  = "jychoi-hpc/ORBIT-2"
cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "orbit2")
os.makedirs(cache, exist_ok=True)

existing = glob.glob(os.path.join(cache, "**/*.ckpt"), recursive=True)
if existing:
    chosen = next((f for f in sorted(existing) if "8m" in f), sorted(existing)[0])
    print(chosen); raise SystemExit(0)

files = [f for f in list_repo_files(repo) if f.endswith(".ckpt") and "pretrain" in f]
if not files:
    files = [f for f in list_repo_files(repo) if f.endswith(".ckpt")]
if not files:
    raise SystemExit("No .ckpt files found on HF")

chosen = next((f for f in sorted(files) if "8m" in f), sorted(files)[0])
local = hf_hub_download(repo_id=repo, filename=chosen, local_dir=cache)
print(local)
PYEOF
)
    fi

    # Stamp checkpoint path into config. The bind-mounted /config/config.yaml
    # is read-only (only the file is mounted, not the directory), so copy to
    # a writable location first. Data path was already stamped on the host.
    cp /config/config.yaml /tmp/orbit2_config.yaml
    sed -i "s|__CKPT_PATH__|${CKPT}|g" /tmp/orbit2_config.yaml
    export ORBIT2_CONFIG_PATH=/tmp/orbit2_config.yaml
    echo "--- Config ready: $ORBIT2_CONFIG_PATH ---"
fi

# --- Distributed inference (Phase 2) ---
# Write a per-rank wrapper that maps torchrun env vars to SLURM env vars
# (upstream visualize.py reads SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS)
RANK_WRAPPER=/tmp/orbit2_rank_wrapper.py
cat > "$RANK_WRAPPER" << 'WRAPPER_PYEOF'
"""Rank wrapper: maps torchrun env vars to SLURM env vars for upstream compat."""
import os, sys

os.environ["SLURM_NTASKS"]  = os.environ.get("WORLD_SIZE", "8")
os.environ["SLURM_PROCID"]  = os.environ.get("RANK", "0")
os.environ["SLURM_LOCALID"] = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("HOSTNAME", os.environ.get("MASTER_ADDR", "localhost"))

rank = os.environ["SLURM_PROCID"]
ngpu = os.environ["SLURM_NTASKS"]
print(f"[rank {rank} / {ngpu}] GPU {os.environ['SLURM_LOCALID']} on {os.uname().nodename}")

os.environ["PYTHONPATH"] = "/tmp/mpi4py_stub:/orbit2/src:/orbit2:" + os.environ.get("PYTHONPATH", "")
os.chdir("/orbit2/examples")

sys.argv = [
    sys.argv[0],
    os.environ.get("ORBIT2_CONFIG_PATH", "/config/config.yaml"),
    "--index", "0",
    "--variable", "total_precipitation_24hr",
    "--master-port", os.environ.get("MASTER_PORT", "29500"),
]

# exec into our run_visualize.py wrapper which then execs upstream visualize.py
exec(open("/examples/run_visualize.py").read())
WRAPPER_PYEOF

EXTRA_ARGS=""
if [[ "$USE_SYNTHETIC" != "1" ]] && [[ -n "$CKPT" ]]; then
    EXTRA_ARGS="--checkpoint $CKPT"
fi

echo ""
echo "--- Phase 2: ${NPROC}-GPU distributed inference via torchrun ---"
torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    "$RANK_WRAPPER" $EXTRA_ARGS

echo "=== ORBIT-2 inference complete ==="
RUNEOF
chmod +x "$RUN_SCRIPT"

# ---------------------------------------------------------------------------
# Resolve config bind-mount
# ---------------------------------------------------------------------------
CONFIG_ABS="$(realpath "$CONFIG")"

# Volume mounts
VOLUMES=(
    -v "$ORBIT2_ROOT":/orbit2
    -v "$SCRIPT_DIR":/examples:ro
    -v "$(dirname "$RUN_SCRIPT")":"$(dirname "$RUN_SCRIPT")"
    -v "${CONFIG_ABS}":/config/config.yaml
    -v "$MIOPEN_HOST":/tmp/miopen
)
if [[ -n "${SYNTH_DIR}" ]]; then
    VOLUMES+=(-v "$SYNTH_DIR":/synth_data)
fi

# ---------------------------------------------------------------------------
# Cleanup trap — remove container on exit
# ---------------------------------------------------------------------------
cleanup() {
    echo "--- Cleaning up container $CONTAINER_NAME ---"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    rm -rf "$MIOPEN_HOST" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
docker run --rm \
    "${GPU_ARGS[@]}" \
    --name "$CONTAINER_NAME" \
    --network host \
    --shm-size=16g \
    "${VOLUMES[@]}" \
    -e ORBIT2_ROOT=/orbit2 \
    -e PYTHONNOUSERSITE=1 \
    -e MIOPEN_USER_DB_PATH=/tmp/miopen \
    -e MIOPEN_DISABLE_CACHE="${MIOPEN_DISABLE_CACHE:-1}" \
    -e MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX="${ORBIT2_MIOPEN_WINOGRAD_MPASS_WS_MAX:--1}" \
    -e MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX="${ORBIT2_MIOPEN_MP_BD_WINOGRAD_WS_MAX:--1}" \
    -e MIOPEN_DEBUG_CONV_WINOGRAD="${ORBIT2_MIOPEN_CONV_WINOGRAD:-0}" \
    -e HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}" \
    "$ORBIT2_IMAGE" \
    bash "$RUN_SCRIPT" \
        "$ROCM_WHL_TAG" \
        "$ORBIT2_NPROC" \
        "${ORBIT2_USE_SYNTHETIC:-0}" \
        "${CKPT:-}"

echo ""
echo "=== Done ==="
