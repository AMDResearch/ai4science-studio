#!/usr/bin/env bash
# Build a persistent Apptainer ext3 overlay pre-loaded with all ORBIT-2 pip
# dependencies.  Run once per cluster; reuse the overlay image across jobs to
# skip the ~15-minute pip install phase on every submission.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export ORBIT2_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
#   sbatch build_overlay_amd.sh
#   # then in your inference job:
#   export ORBIT2_OVERLAY=${AI4S_SHARED_DIR:-/your/shared/dir}/models/ORBIT-2/overlays/orbit2-overlay.img
#   sbatch sbatch_infer_amd.sh
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   This script is parameterized by ROCM_WHL_TAG, not by GPU model.  Change
#   ORBIT2_SIF and ROCM_WHL_TAG together to target a different ROCm version.
#
#   Tested image  : rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
#   Covers        : MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950) — all in rocm7.2.2
#   Older hardware: MI100 (gfx908) — use a rocm6.x image and ROCM_WHL_TAG=rocm6.x
#   Future hardware: update ORBIT2_SIF and ROCM_WHL_TAG to match the new image
#
# ── Why this script is non-trivial (read before modifying) ───────────────────
#   pip install --target <overlay-dir> resolves packages completely fresh —
#   it does NOT see what is already installed in the SIF's /opt/venv.  When
#   ORBIT-2's deps declare torch as a dependency, pip downloads a new ROCm
#   torch wheel (~6 GB download / ~3 GB extracted).  This fills the overlay
#   before other packages can be written.
#
#   Solution — NFS staging:
#     1. Install everything into STAGE_DIR (any path outside the overlay with
#        ~6 GB free — NFS, local scratch, or a large /tmp all work).
#     2. Strip torch, torchvision, torchaudio, nvidia-*, and triton from the
#        stage dir (they live in the SIF's venv and are supplied at runtime).
#     3. Copy the stripped ~2 GB of packages into the overlay.
#   The overlay never needs to hold the full torch installation.
#
#   Alternative for clusters with large overlays:
#     Set ORBIT2_OVERLAY_SIZE_MB to 8192 or larger, and set
#     ORBIT2_STAGE_DIR="" (empty).  The script will install directly into the
#     overlay and strip torch afterward.  Simpler, but requires ~7 GB overlay.
#
# ── Why the overlay is built on $TMPDIR ──────────────────────────────────────
#   The overlay image is an ext3 filesystem served via FUSE.  Writing thousands
#   of small Python package files through FUSE into an image that lives on NFS
#   is extremely slow (ext3-on-NFS via FUSE).  We instead:
#     1. Create and populate the ext3 image on $TMPDIR (node-local fast disk).
#     2. Copy the finished image file (one large sequential write) to the NFS
#        destination ($ORBIT2_OVERLAY).
#   This makes the file-intensive install step fast and leaves a single large
#   file on NFS.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   ORBIT2_SIF              Path to the Apptainer SIF image (required)
#   ORBIT2_OVERLAY          Output overlay path
#                           (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/ORBIT-2/overlays/orbit2-overlay.img)
#   ORBIT2_STAGE_DIR        Writable scratch dir for staging install (~6 GB)
#                           Set to "" to install directly into overlay instead
#                           (requires ORBIT2_OVERLAY_SIZE_MB >= 8192)
#                           (default: <overlay-dir>/orbit2-stage)
#   ORBIT2_OVERLAY_SIZE_MB  Size of the ext3 overlay in MB
#                           (default: 7168 with staging, 8192 without)
#   ROCM_IMAGE              Docker image used to build the SIF, used only in
#                           comments — the SIF itself determines the ROCm ver
#   ROCM_WHL_TAG            PyTorch wheel index suffix, e.g. rocm7.2 or rocm8.0
#                           Must match the ROCm version inside the SIF
#                           (default: rocm7.2)

# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#SBATCH --job-name=orbit2-overlay
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=orbit2-overlay-build-%j.out
#SBATCH --error=orbit2-overlay-build-%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROCM_WHL_TAG="${ROCM_WHL_TAG:-rocm7.2}"

if [[ -z "${ORBIT2_SIF:-}" ]]; then
  echo "error: set ORBIT2_SIF to your Apptainer SIF path before submitting" >&2
  echo "  Recommended image: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0" >&2
  exit 1
fi

ORBIT2_BASE="${AI4S_SHARED_DIR:-/your/shared/dir}/models/ORBIT-2"
OVERLAY="${ORBIT2_OVERLAY:-${ORBIT2_BASE}/overlays/orbit2-overlay.img}"

# Staging mode (default): install to NFS scratch, strip torch, copy to overlay.
# Direct mode (ORBIT2_STAGE_DIR=""): install straight into overlay (needs ≥8 GB).
USE_STAGING=true
if [[ -v ORBIT2_STAGE_DIR && -z "${ORBIT2_STAGE_DIR}" ]]; then
  USE_STAGING=false
fi
STAGE_DIR="${ORBIT2_STAGE_DIR:-${TMPDIR:-/tmp}/orbit2-stage}"

if [[ "$USE_STAGING" == true ]]; then
  OVERLAY_SIZE_MB="${ORBIT2_OVERLAY_SIZE_MB:-7168}"
else
  OVERLAY_SIZE_MB="${ORBIT2_OVERLAY_SIZE_MB:-8192}"
fi

# Build overlay on local disk; copy finished image to NFS destination.
LOCAL_OVERLAY="${TMPDIR:-/tmp}/orbit2-overlay-${SLURM_JOB_ID:-$$}.img"

echo "=== ORBIT-2 overlay build ==="
echo "  SIF          : $ORBIT2_SIF"
echo "  Overlay (NFS): $OVERLAY  (${OVERLAY_SIZE_MB} MB)"
echo "  Build on     : $LOCAL_OVERLAY  (node-local $TMPDIR)"
echo "  ROCm whl tag : $ROCM_WHL_TAG"
echo "  Staging mode : $USE_STAGING  (stage dir: ${STAGE_DIR:-n/a})"
echo "  Node         : $(hostname)  Date: $(date)"
echo ""

if [[ -f "$OVERLAY" ]]; then
  echo "Overlay already exists: $OVERLAY"
  echo "Delete it to rebuild: rm $OVERLAY"
  exit 0
fi

# ---------------------------------------------------------------------------
# Write the install script that runs inside the container
# ---------------------------------------------------------------------------
SCRIPTS_DIR="${TMPDIR:-/tmp}/orbit2-overlay-scripts"
mkdir -p "$SCRIPTS_DIR"

cat > "$SCRIPTS_DIR/overlay_install.sh" << INNEREOF
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

PKG=/opt/orbit2-pkgs
ROCM_WHL_TAG="${ROCM_WHL_TAG}"
USE_STAGING="${USE_STAGING}"
STAGE_DIR="${STAGE_DIR:-}"

# ---------------------------------------------------------------------------
# Determine install target: staging dir (NFS) or overlay dir directly
# ---------------------------------------------------------------------------
if [[ "\$USE_STAGING" == true ]]; then
  rm -rf "\$STAGE_DIR" && mkdir -p "\$STAGE_DIR"
  INSTALL_TARGET="\$STAGE_DIR"
  echo "--- Installing to NFS staging dir: \$STAGE_DIR ---"
else
  mkdir -p "\$PKG"
  INSTALL_TARGET="\$PKG"
  echo "--- Installing directly into overlay: \$PKG ---"
fi

# ---------------------------------------------------------------------------
# Install deps via --extra-index-url so pip picks the ROCm torch wheel.
# The ROCm torch wheel has NO nvidia-* co-package dependencies, so the entire
# CUDA package chain (nvidia-cublas, nvidia-cudnn, nvidia-nccl, etc.) is never
# resolved.  We then strip torch from the install target because the SIF's
# /opt/venv already contains the correct ROCm torch at runtime.
# ---------------------------------------------------------------------------
echo "--- Step 1: ORBIT-2 core deps (mpi4py, huggingface-hub, pytorch-lightning) ---"
pip install -q --no-cache-dir --target "\$INSTALL_TARGET" \
    --extra-index-url https://download.pytorch.org/whl/\${ROCM_WHL_TAG} \
    mpi4py huggingface-hub 2>&1 | tail -3

# pytorch-lightning declares torch as a dependency; --no-deps prevents pulling
# a CUDA torch.  lightning-utilities and torchmetrics have no torch dep.
pip install -q --no-cache-dir --no-deps --target "\$INSTALL_TARGET" \
    pytorch-lightning 2>&1 | tail -3
pip install -q --no-cache-dir --target "\$INSTALL_TARGET" \
    --extra-index-url https://download.pytorch.org/whl/\${ROCM_WHL_TAG} \
    lightning-utilities torchmetrics 2>&1 | tail -3

echo "--- Step 2: science / data deps ---"
pip install -q --no-cache-dir --target "\$INSTALL_TARGET" \
    --extra-index-url https://download.pytorch.org/whl/\${ROCM_WHL_TAG} \
    "timm==0.9.2" "tensorboard==2.11.2" wandb \
    cdsapi "dask>=2022.2.0" "importlib-metadata==4.13.0" \
    "matplotlib>=3.5.3" "netcdf4>=1.6.2" "scikit-learn>=1.0.2" \
    "xarray>=0.20.2" "rasterio>=1.3.7" scikit-image einops lpips \
    pyyaml 2>&1 | tail -5

echo "--- Step 3: xformers for ROCm (--no-deps to avoid CUDA torchvision) ---"
pip install -q --no-cache-dir --no-deps -U xformers \
    --index-url https://download.pytorch.org/whl/\${ROCM_WHL_TAG} \
    --target "\$INSTALL_TARGET" 2>&1 | tail -3

echo "--- Step 4: xformers.components shim (removed in xformers >=0.0.28) ---"
python3 - "\$INSTALL_TARGET" << 'PYEOF'
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
print("xformers.components shim written to", xf)
PYEOF

echo "--- Step 5: strip torch / nvidia / triton from install target ---"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
    rm -rf "\${INSTALL_TARGET}/\${pkg}" "\${INSTALL_TARGET}/\${pkg}"-*.dist-info 2>/dev/null || true
done

if [[ "\$USE_STAGING" == true ]]; then
  echo "--- Step 6: copy stripped packages from staging → overlay ---"
  echo "  Staging size: \$(du -sh "\$STAGE_DIR" | cut -f1)"
  mkdir -p "\$PKG"
  tar -C "\$STAGE_DIR" -cf - . | tar -C "\$PKG" -xf -
  # Final safety strip after copy
  for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
      rm -rf "\${PKG}/\${pkg}" "\${PKG}/\${pkg}"-*.dist-info 2>/dev/null || true
  done
fi

echo "--- Overlay contents: \$(du -sh \$PKG | cut -f1) ---"

echo "--- Verifying ---"
PYTHONPATH="\$PKG" python3 -c "
import xformers, mpi4py, pytorch_lightning, torch
print('xformers         :', xformers.__version__)
print('mpi4py           :', mpi4py.__version__)
print('pytorch_lightning:', pytorch_lightning.__version__)
print('torch            :', torch.__version__)
assert 'rocm' in torch.__version__, 'ERROR: non-ROCm torch in use!'
print('ROCm torch confirmed.')
"
echo "=== Overlay install complete ==="
INNEREOF

chmod +x "$SCRIPTS_DIR/overlay_install.sh"

# ---------------------------------------------------------------------------
# Create overlay on local disk, run install, then copy image to NFS
# ---------------------------------------------------------------------------
echo "Creating ${OVERLAY_SIZE_MB} MB ext3 overlay on local disk: $LOCAL_OVERLAY ..."
apptainer overlay create --size "$OVERLAY_SIZE_MB" "$LOCAL_OVERLAY"

echo "Running install inside container ..."
apptainer exec \
    --rocm \
    --overlay "${LOCAL_OVERLAY}:rw" \
    --bind "$SCRIPTS_DIR":/scripts \
    --bind "${STAGE_DIR}:${STAGE_DIR}" \
    "$ORBIT2_SIF" \
    bash /scripts/overlay_install.sh

echo ""
echo "--- Copying finished overlay to NFS: $OVERLAY ---"
echo "  Image size: $(du -sh "$LOCAL_OVERLAY" | cut -f1)"
cp "$LOCAL_OVERLAY" "$OVERLAY"
rm -f "$LOCAL_OVERLAY"
echo "  Done."

echo ""
echo "=== Overlay ready ==="
ls -lh "$OVERLAY"
echo ""
echo "To use in inference jobs:"
echo "  export ORBIT2_OVERLAY=$OVERLAY"
echo "  sbatch sbatch_infer_amd.sh"
