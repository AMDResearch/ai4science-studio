#!/usr/bin/env bash
# StormCast deterministic inference on AMD Instinct via SLURM + Apptainer.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export SC_SIF=/path/to/rocm_pytorch.sif
#   export SC_START=2025-01-01T06 SC_STEPS=6
#   sbatch sbatch_inference_amd.sh
#
#   # Faster subsequent runs (skip ~5-min pip install):
#   export SC_OVERLAY=/path/to/stormcast-overlay.img   # built once with build_overlay_amd.sh
#   sbatch sbatch_inference_amd.sh
#
# ── Prerequisites ─────────────────────────────────────────────────────────────
#   1. Pull the ROCm PyTorch container and build an Apptainer SIF:
#        apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
#      Recommended image: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
#   2. (Optional) Build a persistent overlay with pre-installed deps:
#        sbatch build_overlay_amd.sh   # runs once, creates stormcast-overlay.img
#      Without an overlay, deps are installed into a per-job temp dir on the
#      host and bind-mounted into the container (~5 min overhead per job).
#   3. Adjust #SBATCH directives below for your site's partition and account.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   SC_SIF        Path to Apptainer SIF image
#                 (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/images/rocm_pytorch.sif)
#   SC_OVERLAY    Path to pre-built ext3 overlay (optional, skips pip install).
#                 (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/StormCast/overlays/stormcast-overlay.img)
#                 If unset (or file missing), deps are installed into a
#                 per-job temp dir at startup (~5 min overhead).
#   SC_START      Forecast start time ISO-8601, e.g. 2025-01-01T06 (default: 2025-01-01T06)
#   SC_STEPS      Number of 1-h inference steps (default: 6)
#   SC_OUTPUT     Output zarr path
#                 (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/StormCast/outputs/pred-<start>.zarr)
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   rocm7.2.2 image covers MI250X (gfx90a), MI300X (gfx942), and MI350X (gfx950).
#   For older hardware (MI100/gfx908): use a rocm6.x image.
#   For future hardware: update SC_SIF to the matching ROCm image.
#
# See: ../recipes/inference/README.md

# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#SBATCH --job-name=stormcast-infer
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=stormcast-infer-%j.out
#SBATCH --error=stormcast-infer-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

# ---------------------------------------------------------------------------
# Configuration (override by exporting before sbatch)
# ---------------------------------------------------------------------------
SC_START="${SC_START:-2025-01-01T06}"
SC_STEPS="${SC_STEPS:-6}"
SC_OUTPUT="${SC_OUTPUT:-}"          # leave empty for default outputs/pred-<date>.zarr

# Apptainer / bare-metal selection
SC_BASE="${AI4S_SHARED_DIR:-/your/shared/dir}/models/StormCast"
SC_SIF="${SC_SIF:-${AI4S_SHARED_DIR:-/your/shared/dir}/images/rocm_pytorch.sif}"

# Optional pre-built overlay (skips pip install on each job)
SC_OVERLAY="${SC_OVERLAY:-${SC_BASE}/overlays/stormcast-overlay.img}"

# ---------------------------------------------------------------------------
# Resolve output path and wipe any previous run's zarr
# (ZarrBackend refuses to overwrite an existing store)
# ---------------------------------------------------------------------------
if [[ -z "${SC_OUTPUT}" ]]; then
    SAFE_START="${SC_START//[: T]/-}"
    SC_OUTPUT="${SC_BASE}/outputs/pred-${SAFE_START}.zarr"
fi
echo "  Output zarr : ${SC_OUTPUT}"
if [[ -e "${SC_OUTPUT}" ]]; then
    echo "  Removing previous output: ${SC_OUTPUT}"
    rm -rf "${SC_OUTPUT}"
fi

# Build argument list
EXTRA_ARGS=(--output "${SC_OUTPUT}")

echo "=== StormCast deterministic inference ==="
echo "  Start : ${SC_START}"
echo "  Steps : ${SC_STEPS}"
echo "  Job   : ${SLURM_JOB_ID:-local}"

# ---------------------------------------------------------------------------
# Apptainer path
# ---------------------------------------------------------------------------
if [[ -n "${SC_SIF}" ]]; then
    echo "  Runtime : Apptainer (${SC_SIF})"

    OVERLAY_ARG=()
    PKGDIR_BIND=()
    if [[ -n "${SC_OVERLAY}" ]] && [[ -f "${SC_OVERLAY}" ]]; then
        OVERLAY_ARG=(--overlay "${SC_OVERLAY}:ro")
        echo "  Overlay : ${SC_OVERLAY}"
    else
        # ---------------------------------------------------------------
        # No-overlay dep install (~5 min one-time per job)
        # Install into a per-job host dir, bind-mount to /opt/stormcast-pkgs.
        # Uses --extra-index-url so pip resolves torch from the ROCm wheel
        # index (no nvidia-* CUDA co-deps). Torch is stripped afterward.
        # This is more robust than --no-deps, which also drops transitive
        # runtime deps (pyvers, treelib, etc.).
        # ---------------------------------------------------------------
        SC_PKGDIR="${TMPDIR:-/tmp}/stormcast-pkgs-${SLURM_JOB_ID:-$$}"
        mkdir -p "$SC_PKGDIR"
        PKGDIR_BIND=(--bind "${SC_PKGDIR}:/opt/stormcast-pkgs")
        ROCM_WHL_TAG="${ROCM_WHL_TAG:-rocm7.2}"
        echo "  No overlay — installing deps → $SC_PKGDIR (~5 min)"

        _INSTALL_SCRIPT="${TMPDIR:-/tmp}/sc_dep_install_${SLURM_JOB_ID:-$$}.sh"
        cat > "$_INSTALL_SCRIPT" << 'INSTALLEOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate
PKG=/opt/stormcast-pkgs
WHL="https://download.pytorch.org/whl/${ROCM_WHL_TAG}"

echo "--- [1/2] Installing earth2studio[stormcast] + cartopy ---"
pip install -q --no-cache-dir --target "$PKG" \
    --extra-index-url "$WHL" \
    "earth2studio[stormcast]" cartopy 2>&1 | tail -5

echo "--- [2/2] Strip torch / nvidia / triton (keep SIF ROCm torch) ---"
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
            "$SC_SIF" \
            bash "$_INSTALL_SCRIPT"

        echo "  Dep install complete"
    fi

    GPU_OK=$(apptainer exec --rocm "${SC_SIF}" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$GPU_OK" != "True" ]]; then
        echo "WARNING: torch.cuda.is_available() = False — falling back to CPU." >&2
        echo "  Try a newer ROCm SIF or set HSA_OVERRIDE_GFX_VERSION=9.4.2" >&2
    fi

    apptainer exec \
        --rocm \
        "${OVERLAY_ARG[@]}" "${PKGDIR_BIND[@]}" \
        --bind "${SCRIPT_DIR}:/workspace" \
        --env LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib \
        "${SC_SIF}" \
        bash -c '
source /opt/venv/bin/activate
export PYTHONPATH="/opt/stormcast-pkgs:${PYTHONPATH:-}"

python /workspace/run_inference.py \
    --start "'"${SC_START}"'" \
    --steps "'"${SC_STEPS}"'" \
    '"${EXTRA_ARGS[*]}"'
'

# ---------------------------------------------------------------------------
# Bare-metal / module path
# ---------------------------------------------------------------------------
else
    echo "  Runtime: bare-metal (activate your env before submitting)"
    # Uncomment whichever applies to your site:
    # source ~/miniconda3/etc/profile.d/conda.sh && conda activate stormcast
    # module load rocm
    python "${SCRIPT_DIR}/run_inference.py" \
        --start "${SC_START}" \
        --steps "${SC_STEPS}" \
        "${EXTRA_ARGS[@]}"
fi

echo "=== Done ==="
echo "  Output: ${SC_OUTPUT}"
