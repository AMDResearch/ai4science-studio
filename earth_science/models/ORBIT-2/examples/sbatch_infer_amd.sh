#!/usr/bin/env bash
# ORBIT-2 visualization / inference-style run on AMD Instinct via SLURM.
#
# Works in two modes, selected by whether ORBIT2_SIF is set:
#   Apptainer mode  — set ORBIT2_SIF to a ROCm PyTorch SIF image path.
#                     No conda/module setup required; container is self-contained.
#   Bare-metal mode — ORBIT2_SIF unset; activate your conda/module env before
#                     submitting (uncomment the activation lines below).
#
# Quick-start with synthetic data (no real ERA5/PRISM data required):
#   export ORBIT2_ROOT=/path/to/ORBIT-2-clone
#   export ORBIT2_SIF=/path/to/rocm_pytorch.sif   # optional but recommended
#   export ORBIT2_USE_SYNTHETIC=1
#   sbatch sbatch_infer_amd.sh
#
# Prerequisites (all modes):
#   1. Clone https://github.com/XiaoWang-Github/ORBIT-2 into ORBIT2_ROOT.
#   2. Download a pretrain checkpoint from https://huggingface.co/jychoi-hpc/ORBIT-2
#      and set ORBIT2_CHECKPOINT to its path.
#
# Prerequisites (bare-metal mode only):
#   3. Install ORBIT-2 per upstream README (AMD + ROCm).
#   4. Set low_res_dir / high_res_dir in your YAML to your data paths,
#      or set ORBIT2_USE_SYNTHETIC=1 to auto-generate a small synthetic dataset.
#
# GPU / ROCm compatibility:
#   Tested image: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
#   Covers: MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950) — all in rocm7.2.2
#   For older hardware (MI100/gfx908): use a rocm6.x image.
#
# Key environment variables:
#   ORBIT2_ROOT           Path to ORBIT-2 clone (required)
#   ORBIT2_CHECKPOINT     Path to .ckpt file (required unless ORBIT2_USE_SYNTHETIC=1,
#                         in which case it is downloaded from HF automatically)
#   ORBIT2_CONFIG         Config YAML basename (default: interm_8m_ft.yaml).
#                         Ignored when ORBIT2_USE_SYNTHETIC=1.
#   ORBIT2_USE_SYNTHETIC  Set to 1 to use make_synthetic_data.py + the bundled
#                         interm_8m_synthetic.yaml instead of real data.
#                         The checkpoint is still downloaded from HF if
#                         ORBIT2_CHECKPOINT is not set.
#   ORBIT2_SIF            Path to Apptainer SIF image for the ROCm PyTorch container.
#                         If unset, runs bare-metal.
#   ORBIT2_OVERLAY        Path to a pre-built ext3 overlay image (optional).
#                         Build once with build_overlay_amd.sh, then reuse across jobs.
#   STUDIO_ORBIT2_LAUNCHER  Override path to run_visualize.py (rarely needed).
#
# Multi-node scaling:
#   Increase --nodes and --ntasks proportionally (8 tasks per node typical).
#   The srun command is identical — SLURM handles rank assignment.
#
# See ../recipes/inference-and-visualization.md and ../recipes/local-cluster-amd.md

#SBATCH -A YOUR_PROJECT_HERE
#SBATCH -J orbit2-vis
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -o orbit2-vis-%j.out
#SBATCH -e orbit2-vis-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
LAUNCHER="${STUDIO_ORBIT2_LAUNCHER:-$SCRIPT_DIR/run_visualize.py}"

if [[ -z "${ORBIT2_ROOT:-}" ]]; then
  echo "error: export ORBIT2_ROOT to your ORBIT-2 clone path" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Bare-metal environment setup (Apptainer mode skips this block)
# ---------------------------------------------------------------------------
# Uncomment whichever applies to your site:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate orbit
# module load rocm

export PYTHONNOUSERSITE=1
export HSA_NO_SCRATCH_RECLAIM=1
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/orbit2-miopen-${SLURM_JOB_ID:-$$}"
mkdir -p "$MIOPEN_USER_DB_PATH"

# ---------------------------------------------------------------------------
# Synthetic data (optional — generates ~2 MB dataset, no ERA5/PRISM required)
# ---------------------------------------------------------------------------
if [[ "${ORBIT2_USE_SYNTHETIC:-0}" == "1" ]]; then
  SYNTH_DIR="${TMPDIR:-/tmp}/orbit2-synthetic-${SLURM_JOB_ID:-$$}"
  echo "--- Generating synthetic ORBIT-2 dataset → $SYNTH_DIR ---"
  python "$SCRIPT_DIR/make_synthetic_data.py" --out-dir "$SYNTH_DIR"

  # Stamp data and checkpoint paths into a temp copy of the synthetic config
  SYNTH_CONFIG="${TMPDIR:-/tmp}/interm_8m_synthetic_${SLURM_JOB_ID:-$$}.yaml"
  sed "s|__SYNTH_DATA_DIR__|$SYNTH_DIR|g" \
      "$SCRIPT_DIR/interm_8m_synthetic.yaml" > "$SYNTH_CONFIG"

  CONFIG="$SYNTH_CONFIG"

  if [[ -n "${ORBIT2_CHECKPOINT:-}" ]]; then
    CKPT="$ORBIT2_CHECKPOINT"
  else
    echo "--- ORBIT2_CHECKPOINT not set; downloading smallest pretrain ckpt from HF ---"
    HF_DL_SCRIPT=$(cat <<'PYEOF'
from huggingface_hub import list_repo_files, hf_hub_download
import os, glob, tempfile

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
    if [[ -n "${ORBIT2_SIF:-}" ]]; then
      OVERLAY_ARG=()
      if [[ -n "${ORBIT2_OVERLAY:-}" ]] && [[ -f "${ORBIT2_OVERLAY}" ]]; then
        OVERLAY_ARG=(--overlay "${ORBIT2_OVERLAY}:ro")
      fi
      CKPT=$(apptainer exec "${OVERLAY_ARG[@]}" \
          --env PYTHONPATH=/opt/orbit2-pkgs \
          "$ORBIT2_SIF" \
          bash -c "source /opt/venv/bin/activate && PYTHONPATH=/opt/orbit2-pkgs python3 -c \"\$1\"" _ "$HF_DL_SCRIPT")
    else
      CKPT=$(python - <<< "$HF_DL_SCRIPT")
    fi
  fi

  # Stamp checkpoint path into the config
  sed -i "s|__CKPT_PATH__|$CKPT|g" "$CONFIG"
  echo "--- Synthetic config ready: $CONFIG ---"
else
  CONFIG="${ORBIT2_CONFIG:-interm_8m_ft.yaml}"
fi

# ---------------------------------------------------------------------------
# Checkpoint arg (real-data mode; synthetic mode embeds it in the YAML)
# ---------------------------------------------------------------------------
EXTRA=()
if [[ "${ORBIT2_USE_SYNTHETIC:-0}" != "1" ]] && [[ -n "${ORBIT2_CHECKPOINT:-}" ]]; then
  EXTRA+=(--checkpoint "$ORBIT2_CHECKPOINT")
fi

# Optional: Slingshot / RCCL tuning on Cray systems only.
# export FI_MR_CACHE_MONITOR=kdreg2

# ---------------------------------------------------------------------------
# Apptainer mode
# ---------------------------------------------------------------------------
if [[ -n "${ORBIT2_SIF:-}" ]]; then
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
  MASTER_PORT="${ORBIT2_MASTER_PORT:-29500}"

  OVERLAY_ARG=()
  if [[ -n "${ORBIT2_OVERLAY:-}" ]] && [[ -f "${ORBIT2_OVERLAY}" ]]; then
    OVERLAY_ARG=(--overlay "${ORBIT2_OVERLAY}:ro")
    echo "--- Using overlay: ${ORBIT2_OVERLAY} ---"
  fi

  # Write a per-rank launcher script so srun can invoke it directly.
  RANK_SCRIPT="${TMPDIR:-/tmp}/orbit2_rank_${SLURM_JOB_ID:-$$}.sh"
  cat > "$RANK_SCRIPT" << RANKEOF
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate
echo "[rank \$SLURM_PROCID / \$SLURM_NTASKS] GPU \$SLURM_LOCALID on \$(hostname)"
export PYTHONPATH="/opt/orbit2-pkgs:/orbit2/src:/orbit2:\${PYTHONPATH:-}"
cd /orbit2/examples
python3 /examples/run_visualize.py \\
    /config/config.yaml \\
    --index 0 --variable total_precipitation_24hr \\
    --master-port \$MASTER_PORT \\
    ${EXTRA[@]:+${EXTRA[@]}}
RANKEOF
  chmod +x "$RANK_SCRIPT"

  # Bind the config into /config/config.yaml inside the container
  CONFIG_BIND="$(realpath "$CONFIG"):/config/config.yaml"

  echo "ORBIT2_SIF=$ORBIT2_SIF  MASTER_ADDR=$MASTER_ADDR  SLURM_NTASKS=${SLURM_NTASKS:-?}"
  echo ""
  echo "--- Phase 1: generate synthetic data (single container, before srun) ---"
  if [[ "${ORBIT2_USE_SYNTHETIC:-0}" == "1" ]]; then
    apptainer exec \
        --rocm \
        "${OVERLAY_ARG[@]}" \
        --bind "$SCRIPT_DIR":/examples \
        "$ORBIT2_SIF" \
        python3 /examples/make_synthetic_data.py --out-dir "$SYNTH_DIR"
  fi

  echo ""
  echo "--- Phase 2: ${SLURM_NTASKS:-8}-way distributed visualize (srun --mpi=pmix) ---"
  # --mpi=pmix: provides a PMIx v4 server that the container's OpenMPI/mpi4py
  # can connect to. Default srun MPI modes (including --mpi=none) fail because
  # the container's mpi4py auto-calls MPI_Init and cannot reach the PMIx server
  # through the container namespace.
  # HOSTNAME is injected as MASTER_ADDR because visualize.py does:
  #   os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
  # Using localhost only works single-node; the scontrol-derived hostname works
  # for any number of nodes without changing the srun command.
  srun --mpi=pmix apptainer exec \
      --rocm \
      "${OVERLAY_ARG[@]}" \
      --bind "$ORBIT2_ROOT":/orbit2 \
      --bind "$SCRIPT_DIR":/examples \
      --bind "$(dirname "$RANK_SCRIPT"):$(dirname "$RANK_SCRIPT")" \
      --bind "$CONFIG_BIND" \
      --env HOSTNAME="$MASTER_ADDR" \
      --env MASTER_PORT="$MASTER_PORT" \
      --env PYTHONPATH="/opt/orbit2-pkgs:/orbit2/src:/orbit2" \
      --env LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib \
      "$ORBIT2_SIF" \
      bash "$RANK_SCRIPT"

# ---------------------------------------------------------------------------
# Bare-metal mode
# ---------------------------------------------------------------------------
else
  echo "LAUNCHER=$LAUNCHER ORBIT2_ROOT=$ORBIT2_ROOT CONFIG=$CONFIG SLURM_NTASKS=${SLURM_NTASKS:-unset}"
  srun "${PYTHON:-python}" "$LAUNCHER" \
      --orbit2-root "$ORBIT2_ROOT" "$CONFIG" "${EXTRA[@]}"
fi
