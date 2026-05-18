#!/usr/bin/env bash
# Build a persistent Apptainer ext3 overlay pre-loaded with all HydraGNN pip
# dependencies.  Run once per cluster; reuse the overlay image across jobs to
# skip the pip install phase on every submission.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export HG_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
#   sbatch build_overlay_amd.sh
#   # then in your inference job:
#   export HG_OVERLAY=/path/to/hydragnn-overlay.img
#   sbatch sbatch_infer_amd.sh
#
# ── torch-scatter / torch-sparse / torch-cluster / torch-spline-conv ─────────
#   No official ROCm wheels exist at data.pyg.org.  We use pre-built ROCm
#   wheels from https://github.com/Looong01/pyg-rocm-build (release 15):
#     - PyTorch 2.10.x + ROCm 7.1 wheels run fine on ROCm 7.2.x (ABI compat)
#     - Python 3.12, linux_x86_64
#   torch-geometric (pure Python) is installed from PyPI as normal.
#
# ── Why staging is needed ────────────────────────────────────────────────────
#   pip install --target does not see the SIF's /opt/venv, so torch gets
#   re-downloaded (~6 GB) and fills the overlay.  We stage into a writable
#   scratch dir, strip torch/nvidia/triton, then copy ~1-2 GB into the overlay.
#
# ── Why the overlay is built on $TMPDIR ──────────────────────────────────────
#   The overlay image is an ext3 filesystem served via FUSE.  Writing thousands
#   of small Python package files through FUSE into an image that lives on NFS
#   is extremely slow (ext3-on-NFS via FUSE).  We instead:
#     1. Create and populate the ext3 image on $TMPDIR (node-local fast disk).
#     2. Copy the finished image file (one large sequential write) to the NFS
#        destination ($HG_OVERLAY).
#   This makes the file-intensive install step fast and leaves a single large
#   file on NFS.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   HG_SIF              Path to the Apptainer SIF image (required)
#   HG_OVERLAY          Output overlay path
#                       (default: <sif-dir>/hydragnn-overlay.img)
#   HG_STAGE_DIR        Writable scratch dir for staging install (~6 GB)
#                       (default: <overlay-dir>/hydragnn-stage)
#   HG_OVERLAY_SIZE_MB  Size of the ext3 overlay in MB (default: 4096)
#   PYG_ROCM_RELEASE    Looong01 release tag to use (default: 15)
#   PYG_PYTHON_TAG      Python tag in zip filename (default: py312)

#SBATCH --job-name=hydragnn-overlay
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=hydragnn-overlay-build-%j.out
#SBATCH --error=hydragnn-overlay-build-%j.out

set -euo pipefail

HG_BASE="${AI4S_SHARED_DIR:-/your/shared/dir}/models/HydraGNN"
HG_SIF="${HG_SIF:-${HG_BASE}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
OVERLAY="${HG_OVERLAY:-${HG_BASE}/overlays/hydragnn-overlay.img}"
STAGE_DIR="${HG_STAGE_DIR:-${HG_BASE}/stage/hydragnn-stage}"
OVERLAY_SIZE_MB="${HG_OVERLAY_SIZE_MB:-4096}"
PYG_ROCM_RELEASE="${PYG_ROCM_RELEASE:-15}"
PYG_PYTHON_TAG="${PYG_PYTHON_TAG:-py312}"

PYG_ZIP_NAME="torch-2.10.0-rocm-7.1-${PYG_PYTHON_TAG}-linux_x86_64.zip"
PYG_ZIP_URL="https://github.com/Looong01/pyg-rocm-build/releases/download/${PYG_ROCM_RELEASE}/${PYG_ZIP_NAME}"
PYG_WHEELS_DIR="${STAGE_DIR}/pyg-rocm-wheels"

# Build the overlay on local disk to avoid ext3-on-NFS small-file I/O penalty;
# copy the finished image file to NFS destination as one sequential write.
LOCAL_OVERLAY="${TMPDIR:-/tmp}/hydragnn-overlay-${SLURM_JOB_ID:-$$}.img"

echo "=== HydraGNN overlay build ==="
echo "  SIF          : $HG_SIF"
echo "  Overlay (NFS): $OVERLAY  (${OVERLAY_SIZE_MB} MB)"
echo "  Build on     : $LOCAL_OVERLAY  (node-local $TMPDIR)"
echo "  Stage dir    : $STAGE_DIR"
echo "  PyG wheels   : Looong01 release ${PYG_ROCM_RELEASE} (torch 2.10, ROCm 7.1, ${PYG_PYTHON_TAG})"
echo "  Node         : $(hostname)  Date: $(date)"
echo ""

if [[ -f "$OVERLAY" ]]; then
  echo "Overlay already exists: $OVERLAY"
  echo "Delete it to rebuild: rm $OVERLAY"
  exit 0
fi

mkdir -p "$STAGE_DIR" "$PYG_WHEELS_DIR"

# ---------------------------------------------------------------------------
# Download and extract Looong01 PyG ROCm wheels (on the host, before container)
# ---------------------------------------------------------------------------
PYG_ZIP="${STAGE_DIR}/${PYG_ZIP_NAME}"
if [[ ! -f "$PYG_ZIP" ]]; then
  echo "--- Downloading PyG ROCm wheels from Looong01 release ${PYG_ROCM_RELEASE} ---"
  curl -fL "$PYG_ZIP_URL" -o "$PYG_ZIP"
else
  echo "--- PyG ROCm zip already downloaded: $PYG_ZIP ---"
fi

echo "--- Extracting wheels ---"
unzip -q -o "$PYG_ZIP" -d "$PYG_WHEELS_DIR"
echo "Wheels extracted:"
ls "$PYG_WHEELS_DIR"/*.whl 2>/dev/null || ls "$PYG_WHEELS_DIR"

# ---------------------------------------------------------------------------
# Write the install script that runs inside the container
# ---------------------------------------------------------------------------
SCRIPTS_DIR="${TMPDIR:-/tmp}/hydragnn-overlay-scripts-${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRIPTS_DIR"

cat > "$SCRIPTS_DIR/overlay_install.sh" << INNEREOF
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

PKG=/opt/hydragnn-pkgs
STAGE="${STAGE_DIR}/pkgs"
PYG_WHEELS="${PYG_WHEELS_DIR}"

rm -rf "\$STAGE" && mkdir -p "\$STAGE"
echo "--- torch: \$(python3 -c 'import torch; print(torch.__version__)') ---"
echo "--- Installing to staging dir: \$STAGE ---"

echo "--- Step 1: PyG ROCm wheels (torch-scatter, torch-sparse, torch-cluster, etc.) ---"
# --no-deps: these wheels depend on torch, which is already in the SIF venv
pip install -q --no-cache-dir --no-deps --target "\$STAGE" "\$PYG_WHEELS"/*.whl
echo "  installed: \$(ls \$STAGE | grep torch | tr '\n' ' ')"

echo "--- Step 2: torch-geometric (pure Python) ---"
pip install -q --no-cache-dir --target "\$STAGE" torch-geometric 2>&1 | tail -3

echo "--- Step 3: HydraGNN runtime deps ---"
pip install -q --no-cache-dir --target "\$STAGE" \
    mpi4py tqdm pyyaml tensorboard matplotlib \
    scikit-learn scipy h5py ase 2>&1 | tail -5

echo "--- Step 4: HydraGNN package (clone + install, no deps) ---"
if [[ ! -d /tmp/HydraGNN-src ]]; then
    git clone -q --depth=1 --branch Predictive_GFM_2024 \
        https://github.com/ORNL/HydraGNN.git /tmp/HydraGNN-src
fi
pip install -q --no-cache-dir --target "\$STAGE" --no-deps /tmp/HydraGNN-src 2>&1 | tail -3

echo "--- Step 5: strip torch / nvidia / triton from staging dir ---"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
    rm -rf "\${STAGE}/\${pkg}" "\${STAGE}/\${pkg}"-*.dist-info 2>/dev/null || true
done
echo "  Staging size after strip: \$(du -sh \$STAGE | cut -f1)"

echo "--- Step 6: copy stripped packages → overlay (local disk, fast) ---"
mkdir -p "\$PKG"
cp -r "\$STAGE/." "\$PKG/"
# Final safety strip
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
    rm -rf "\${PKG}/\${pkg}" "\${PKG}/\${pkg}"-*.dist-info 2>/dev/null || true
done
echo "  Overlay size: \$(du -sh \$PKG | cut -f1)"

echo "--- Step 7: verify ---"
PYTHONPATH="\$PKG" python3 -c "
import torch, torch_scatter, torch_geometric, hydragnn
print('torch         :', torch.__version__)
print('torch_scatter : ok')
print('torch_geometric:', torch_geometric.__version__)
print('hydragnn      : ok')
assert torch.cuda.is_available() or 'rocm' in torch.__version__, 'ROCm torch expected'
print('Verification passed.')
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
    --bind "${PYG_WHEELS_DIR}:${PYG_WHEELS_DIR}" \
    "$HG_SIF" \
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
echo "  export HG_OVERLAY=$OVERLAY"
echo "  sbatch sbatch_infer_amd.sh"
