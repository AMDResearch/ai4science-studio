#!/usr/bin/env bash
# Build a persistent Apptainer ext3 overlay pre-loaded with all StormCast pip
# dependencies.  Run once per cluster; reuse the overlay image across jobs to
# skip the ~5-minute pip install phase on every submission.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export SC_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
#   sbatch build_overlay_amd.sh
#   # then in your inference job:
#   export SC_OVERLAY=/path/to/stormcast-overlay.img
#   sbatch sbatch_inference_amd.sh
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   Parameterized by ROCM_WHL_TAG, not GPU model.  Change SC_SIF and
#   ROCM_WHL_TAG together to target a different ROCm version.
#
#   Tested image  : rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
#   Covers        : MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950) — all in rocm7.2.2
#   Older hardware: MI100 (gfx908) — use a rocm6.x image and ROCM_WHL_TAG=rocm6.x
#   Future hardware: update SC_SIF and ROCM_WHL_TAG to match the new image
#
# ── Why this script uses NFS staging (read before modifying) ─────────────────
#   pip install --target <overlay-dir> resolves packages completely fresh and
#   does NOT see what is in the SIF's /opt/venv.  earth2studio[stormcast]
#   transitively requires torch, which pip downloads as a fresh ROCm wheel
#   (~6 GB download / ~3 GB extracted).  A 4 GB overlay runs out of space.
#
#   Strategy — install to a staging dir outside the overlay (NFS, local
#   scratch, or any path with ~6 GB free), strip torch/nvidia/triton there,
#   then copy the remaining ~1.7 GB into the overlay.
#
#   If your cluster has no shared filesystem and a large-enough local scratch:
#     Set SC_STAGE_DIR to a local path with ≥6 GB free.
#   If you prefer a larger overlay instead of staging:
#     Set SC_STAGE_DIR="" and SC_OVERLAY_SIZE_MB=8192 to install directly.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   SC_SIF              Path to the Apptainer SIF image (required)
#   SC_OVERLAY          Output overlay path
#                       (default: same dir as SIF, stormcast-overlay.img)
#   SC_STAGE_DIR        Writable scratch dir for staging (~6 GB needed)
#                       Set to "" to install directly into overlay instead
#                       (requires SC_OVERLAY_SIZE_MB >= 8192)
#                       (default: <overlay-dir>/stormcast-stage)
#   SC_OVERLAY_SIZE_MB  Size of the ext3 overlay in MB
#                       (default: 4096 with staging, 8192 without)
#   ROCM_WHL_TAG        PyTorch wheel index suffix, e.g. rocm7.2 or rocm8.0
#                       Must match the ROCm version inside the SIF
#                       (default: rocm7.2)

#SBATCH --job-name=stormcast-overlay
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=stormcast-overlay-build-%j.out
#SBATCH --error=stormcast-overlay-build-%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROCM_WHL_TAG="${ROCM_WHL_TAG:-rocm7.2}"

if [[ -z "${SC_SIF:-}" ]]; then
  echo "error: set SC_SIF to your Apptainer SIF path before submitting" >&2
  echo "  Recommended image: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0" >&2
  exit 1
fi

OVERLAY="${SC_OVERLAY:-$(dirname "$SC_SIF")/stormcast-overlay.img}"

USE_STAGING=true
if [[ -v SC_STAGE_DIR && -z "${SC_STAGE_DIR}" ]]; then
  USE_STAGING=false
fi
STAGE_DIR="${SC_STAGE_DIR:-$(dirname "$OVERLAY")/stormcast-stage}"

if [[ "$USE_STAGING" == true ]]; then
  OVERLAY_SIZE_MB="${SC_OVERLAY_SIZE_MB:-4096}"
else
  OVERLAY_SIZE_MB="${SC_OVERLAY_SIZE_MB:-8192}"
fi

echo "=== StormCast overlay build ==="
echo "  SIF          : $SC_SIF"
echo "  Overlay      : $OVERLAY  (${OVERLAY_SIZE_MB} MB)"
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
SCRIPTS_DIR="$(dirname "$OVERLAY")/stormcast-overlay-scripts"
mkdir -p "$SCRIPTS_DIR"

cat > "$SCRIPTS_DIR/overlay_install.sh" << INNEREOF
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

PKG=/opt/stormcast-pkgs
ROCM_WHL_TAG="${ROCM_WHL_TAG}"
USE_STAGING="${USE_STAGING}"
STAGE_DIR="${STAGE_DIR:-}"

# ---------------------------------------------------------------------------
# Determine install target
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
# Install earth2studio[stormcast] + cartopy.
#
# earth2studio[stormcast] requires nvidia-physicsnemo>=2.0, which in turn
# needs timm → torchvision (ROCm version is in /opt/venv, so we use --no-deps
# for packages that would pull a CUDA torchvision) and tensordict (same).
#
# --extra-index-url rocm<X.Y> causes pip to pick the ROCm torch wheel when
# resolving torch, which has no nvidia-* CUDA co-package dependencies.
# After install, torch is stripped so the SIF's /opt/venv torch is used at
# runtime (correct ROCm version, already compiled for the target GPU arch).
# ---------------------------------------------------------------------------
echo "--- Installing earth2studio (no [stormcast] extra to avoid nvidia-physicsnemo pulling torch) ---"
pip install -q --no-cache-dir --no-deps --target "\$INSTALL_TARGET" earth2studio 2>&1 | tail -3

echo "--- Installing nvidia-physicsnemo (needed by stormcast.py; no CUDA co-deps) ---"
pip install -q --no-cache-dir --no-deps --target "\$INSTALL_TARGET" "nvidia-physicsnemo>=2.0" 2>&1 | tail -3

# timm and warp-lang both declare torch/torchvision as deps; --no-deps prevents
# a CUDA torchvision being installed on top of the ROCm one in /opt/venv.
echo "--- Installing timm, warp-lang, tensordict --no-deps ---"
pip install -q --no-cache-dir --no-deps \
    --target "\$INSTALL_TARGET" timm warp-lang tensordict 2>&1 | tail -3

echo "--- Installing earth2studio + physicsnemo runtime deps + cartopy ---"
# These packages do not declare torch as a direct dependency.
pip install -q --no-cache-dir --target "\$INSTALL_TARGET" \
    cftime gcsfs h5netcdf h5py "huggingface-hub>=0.27.0" loguru nest-asyncio \
    "netcdf4<1.7.3,>=1.6.4" "pandas<3.0" pyarrow pygrib python-dotenv rich \
    s3fs "tqdm>=4.65.0" "xarray[parallel]>=2023.1.0" "zarr>=3.1.0" \
    einops scipy omegaconf pyproj cartopy \
    nvtx hydra-core gitpython importlib-metadata jaxtyping \
    safetensors 2>&1 | tail -5

echo "--- Stripping torch / nvidia / triton from install target ---"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
    rm -rf "\${INSTALL_TARGET}/\${pkg}" "\${INSTALL_TARGET}/\${pkg}"-*.dist-info 2>/dev/null || true
done

if [[ "\$USE_STAGING" == true ]]; then
  echo "--- Copying stripped packages from staging → overlay ---"
  echo "  Staging size: \$(du -sh "\$STAGE_DIR" | cut -f1)"
  mkdir -p "\$PKG"
  cp -r "\$STAGE_DIR"/. "\$PKG/"
  # Final safety strip after copy
  for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
      rm -rf "\${PKG}/\${pkg}" "\${PKG}/\${pkg}"-*.dist-info 2>/dev/null || true
  done
fi

echo "--- Overlay contents: \$(du -sh \$PKG | cut -f1) ---"

echo "--- Verifying ---"
PYTHONPATH="\$PKG" python3 -c "
import earth2studio, cartopy, torch
print('earth2studio :', earth2studio.__version__)
print('cartopy      :', cartopy.__version__)
print('torch        :', torch.__version__)
assert 'rocm' in torch.__version__, 'ERROR: non-ROCm torch detected!'
print('ROCm torch confirmed.')
"
echo "=== Overlay install complete ==="
INNEREOF

chmod +x "$SCRIPTS_DIR/overlay_install.sh"

# ---------------------------------------------------------------------------
# Create the overlay image and run the install inside the container
# ---------------------------------------------------------------------------
echo "Creating ${OVERLAY_SIZE_MB} MB ext3 overlay image ..."
apptainer overlay create --size "$OVERLAY_SIZE_MB" "$OVERLAY"

echo "Running install inside container ..."
apptainer exec \
    --rocm \
    --overlay "${OVERLAY}:rw" \
    --bind "$SCRIPTS_DIR":/scripts \
    --bind "$(dirname "$OVERLAY")":/overlay-dir \
    "$SC_SIF" \
    bash /scripts/overlay_install.sh

echo ""
echo "=== Overlay ready ==="
ls -lh "$OVERLAY"
echo ""
echo "To use in inference jobs:"
echo "  export SC_OVERLAY=$OVERLAY"
echo "  sbatch sbatch_inference_amd.sh"
