#!/usr/bin/env bash
# Allocation-neutral ORBIT-2 inference runner.
# Invoked from sbatch_infer_amd.sh or run_infer_perf_amd.sh (never nests sbatch).
#
# Env: ORBIT2_ROOT (required), ORBIT2_SIF/OVERLAY/CHECKPOINT/CONFIG,
#      ORBIT2_USE_SYNTHETIC, AI4S_SHARED_DIR, ORBIT2_MASTER_PORT, ...

set -euo pipefail

# Prefer BASH_SOURCE so nested calls (from run_infer_perf_amd.sh / campaign runner)
# do not resolve SCRIPT_DIR to the outer sbatch Command= path. Fall back to
# scontrol only when this file itself was copied into the SLURM spool.
_SRC="${BASH_SOURCE[0]:-}"
if [[ -n "$_SRC" && "$_SRC" != *"/var/spool/slurmd/"* && "$_SRC" != *"/spool/slurmd/"* ]]; then
  SCRIPT_DIR=$(cd "$(dirname "$_SRC")" && pwd)
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
LAUNCHER="${STUDIO_ORBIT2_LAUNCHER:-$SCRIPT_DIR/run_visualize.py}"

ORBIT2_BASE="${AI4S_SHARED_DIR:-/your/shared/dir}/models/ORBIT-2"
ORBIT2_ROOT="${ORBIT2_ROOT:-${ORBIT2_BASE}/code/ORBIT-2}"
ORBIT2_SIF="${ORBIT2_SIF:-${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
ORBIT2_OVERLAY="${ORBIT2_OVERLAY:-${ORBIT2_BASE}/overlays/orbit2-overlay.img}"

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
# ORNL Frontier-validated MIOpen conv flags (bayes-cast launch/launch_diffusion.sh): disable Winograd,
# unbound multi-pass Winograd workspace. Override to A/B on MI355X (e.g. ORBIT2_MIOPEN_CONV_WINOGRAD=1).
export MIOPEN_DISABLE_CACHE="${MIOPEN_DISABLE_CACHE:-1}"
export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX="${ORBIT2_MIOPEN_WINOGRAD_MPASS_WS_MAX:--1}"
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX="${ORBIT2_MIOPEN_MP_BD_WINOGRAD_WS_MAX:--1}"
export MIOPEN_DEBUG_CONV_WINOGRAD="${ORBIT2_MIOPEN_CONV_WINOGRAD:-0}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"

# ---------------------------------------------------------------------------
# No-overlay dep install (Apptainer only — ~15 min one-time per job)
# When ORBIT2_SIF is set but no overlay is provided, install all Python
# dependencies into a per-job temp dir on the host and bind-mount it into
# the container at /opt/orbit2-pkgs (the same path the overlay provides).
# ---------------------------------------------------------------------------
PKGDIR_BIND=()
if [[ -n "${ORBIT2_SIF:-}" ]] && { [[ -z "${ORBIT2_OVERLAY:-}" ]] || [[ ! -f "${ORBIT2_OVERLAY:-}" ]]; }; then
  ORBIT2_PKGDIR="${TMPDIR:-/tmp}/orbit2-pkgs-${SLURM_JOB_ID:-$$}"
  mkdir -p "$ORBIT2_PKGDIR"
  PKGDIR_BIND=(--bind "${ORBIT2_PKGDIR}:/opt/orbit2-pkgs")
  ROCM_WHL_TAG="${ROCM_WHL_TAG:-rocm7.2}"
  echo "--- No overlay: installing ORBIT-2 deps → $ORBIT2_PKGDIR (~15 min) ---"

  _INSTALL_SCRIPT="${TMPDIR:-/tmp}/orbit2_dep_install_${SLURM_JOB_ID:-$$}.sh"
  cat > "$_INSTALL_SCRIPT" << 'INSTALLEOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate
PKG=/opt/orbit2-pkgs
WHL="https://download.pytorch.org/whl/${ROCM_WHL_TAG}"

echo "--- [1/5] Core deps (mpi4py, huggingface-hub, pytorch-lightning) ---"
pip install -q --no-cache-dir --target "$PKG" \
    --extra-index-url "$WHL" \
    mpi4py huggingface-hub 2>&1 | tail -3
pip install -q --no-cache-dir --no-deps --target "$PKG" \
    pytorch-lightning 2>&1 | tail -3
pip install -q --no-cache-dir --target "$PKG" \
    --extra-index-url "$WHL" \
    lightning-utilities torchmetrics 2>&1 | tail -3

echo "--- [2/5] Science / data deps ---"
pip install -q --no-cache-dir --target "$PKG" \
    --extra-index-url "$WHL" \
    "timm==0.9.2" "tensorboard==2.11.2" wandb \
    cdsapi "dask>=2022.2.0" "importlib-metadata==4.13.0" \
    "matplotlib>=3.5.3" "netcdf4>=1.6.2" "scikit-learn>=1.0.2" \
    "xarray>=0.20.2" "rasterio>=1.3.7" scikit-image einops lpips \
    pyyaml 2>&1 | tail -5

echo "--- [3/5] xformers for ROCm ---"
pip install -q --no-cache-dir --no-deps -U xformers \
    --index-url "$WHL" \
    --target "$PKG" 2>&1 | tail -3

echo "--- [4/5] xformers.components shim ---"
python3 - "$PKG" << 'PYEOF'
import pathlib, sys
xf = pathlib.Path(sys.argv[1]) / "xformers"
comp = xf / "components"; attn = comp / "attention"
for d in [comp, attn]: d.mkdir(exist_ok=True)
(comp / "__init__.py").write_text("")
(attn / "__init__.py").write_text("")
(attn / "core.py").write_text(
    "import torch.nn.functional as F\n\n"
    "def scaled_dot_product_attention(q, k, v, att_mask=None, dropout=0.0):\n"
    "    return F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask, dropout_p=dropout)\n"
)
print("xformers.components shim written")
PYEOF

echo "--- [5/5] Strip torch / nvidia / triton ---"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
    rm -rf "${PKG}/${pkg}" "${PKG}/${pkg}"-*.dist-info 2>/dev/null || true
done

echo "--- Installed: $(du -sh $PKG | cut -f1) ---"
INSTALLEOF
  chmod +x "$_INSTALL_SCRIPT"

  apptainer exec --rocm \
      "${PKGDIR_BIND[@]}" \
      --bind "$(dirname "$_INSTALL_SCRIPT"):$(dirname "$_INSTALL_SCRIPT")" \
      --env ROCM_WHL_TAG="$ROCM_WHL_TAG" \
      "$ORBIT2_SIF" \
      bash "$_INSTALL_SCRIPT"

  echo "--- Dep install complete ---"
fi

# ---------------------------------------------------------------------------
# Synthetic data (optional — generates ~2 MB dataset, no ERA5/PRISM required)
# ---------------------------------------------------------------------------
if [[ "${ORBIT2_USE_SYNTHETIC:-0}" == "1" ]]; then
  SYNTH_DIR="${TMPDIR:-/tmp}/orbit2-synthetic-${SLURM_JOB_ID:-$$}"
  mkdir -p "$SYNTH_DIR"
  echo "--- Generating synthetic ORBIT-2 dataset → $SYNTH_DIR ---"
  if [[ -z "${ORBIT2_SIF:-}" ]]; then
    python3 "$SCRIPT_DIR/make_synthetic_data.py" --out-dir "$SYNTH_DIR"
  fi

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
      CKPT=$(apptainer exec "${OVERLAY_ARG[@]}" "${PKGDIR_BIND[@]}" \
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
  GPU_OK=$(apptainer exec --rocm "${ORBIT2_SIF}" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
  if [[ "$GPU_OK" != "True" ]]; then
      echo "WARNING: torch.cuda.is_available() = False — falling back to CPU." >&2
      echo "  Try a newer ROCm SIF or set HSA_OVERRIDE_GFX_VERSION=9.4.2" >&2
  fi

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
    --orbit2-root /orbit2 \\
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
        "${OVERLAY_ARG[@]}" "${PKGDIR_BIND[@]}" \
        --bind "$SCRIPT_DIR":/examples \
        "$ORBIT2_SIF" \
        python3 /examples/make_synthetic_data.py --out-dir "$SYNTH_DIR"
  fi

  echo ""
  echo "--- Phase 2: ${SLURM_NTASKS:-8}-way distributed visualize (srun --mpi=pmix) ---"
  # --mpi=pmix: provides a PMIx server that the container's OpenMPI/mpi4py
  # can connect to for MPI_Init.
  # HOSTNAME is injected as MASTER_ADDR because visualize.py does:
  #   os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
  # Using localhost only works single-node; the scontrol-derived hostname works
  # for any number of nodes without changing the srun command.
  srun --mpi=pmix apptainer exec \
      --rocm \
      "${OVERLAY_ARG[@]}" "${PKGDIR_BIND[@]}" \
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
