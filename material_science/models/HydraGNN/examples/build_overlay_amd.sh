#!/usr/bin/env bash
# Build a persistent Apptainer ext3 overlay pre-loaded with all HydraGNN pip
# dependencies.  Run once per cluster; reuse the overlay image across jobs to
# skip the pip install phase on every submission.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   # From an interactive compute node (has /opt/ompi with mpicc):
#   export AI4S_SHARED_DIR=/path/to/shared   # set via /init-cluster
#   bash build_overlay_amd.sh
#
# ── MPI-enabled adios2 ──────────────────────────────────────────────────────
#   The compute nodes have /opt/ompi (OpenMPI 5.x with mpicc). We bind-mount
#   it into the container during the build so pip can compile adios2 from
#   source with ADIOS2_USE_MPI=ON.  This is required for parallel dataset I/O.
#
# ── torch-scatter / torch-sparse / torch-cluster / torch-spline-conv ────────
#   No official ROCm wheels exist at data.pyg.org.  We use pre-built ROCm
#   wheels from https://github.com/Looong01/pyg-rocm-build (release 15):
#     - PyTorch 2.10.x + ROCm 7.1 wheels run fine on ROCm 7.2.x (ABI compat)
#     - Python 3.12, linux_x86_64
#   torch-geometric (pure Python) is installed from PyPI as normal.
#
# ── Why staging is needed ────────────────────────────────────────────────────
#   pip install --target does not see the SIF's /opt/venv, so torch gets
#   re-downloaded (~6 GB) and fills the overlay.  We stage into /tmp (node-
#   local fast disk), strip torch/nvidia/triton, then copy into the overlay.
#
# ── Key environment variables ────────────────────────────────────────────────
#   AI4S_SHARED_DIR     Shared filesystem root (required)
#   HG_SIF              Path to the Apptainer SIF image
#   HG_OVERLAY          Output overlay path
#   HG_OVERLAY_SIZE_MB  Size of the ext3 overlay in MB (default: 4096)
#   HG_HYDRAGNN_SHA     Git SHA to pin HydraGNN at (default: current main)
#   PYG_ROCM_RELEASE    Looong01 release tag to use (default: 15)
#   PYG_PYTHON_TAG      Python tag in zip filename (default: py312)

set -euo pipefail

HG_BASE="${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}/models/HydraGNN"
HG_SIF="${HG_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
OVERLAY="${HG_OVERLAY:-${HG_BASE}/overlays/hydragnn-overlay.img}"
OVERLAY_SIZE_MB="${HG_OVERLAY_SIZE_MB:-4096}"
HG_HYDRAGNN_SHA="${HG_HYDRAGNN_SHA:-6c45f1682783e66dc89e9e23009f61716186432b}"
PYG_ROCM_RELEASE="${PYG_ROCM_RELEASE:-15}"
PYG_PYTHON_TAG="${PYG_PYTHON_TAG:-py312}"

PYG_ZIP_NAME="torch-2.10.0-rocm-7.1-${PYG_PYTHON_TAG}-linux_x86_64.zip"
PYG_ZIP_URL="https://github.com/Looong01/pyg-rocm-build/releases/download/${PYG_ROCM_RELEASE}/${PYG_ZIP_NAME}"

# Use node-local fast storage for build I/O (SCRATCH_LOCAL from .cluster-config.yaml)
LOCAL_TMP="${SCRATCH_LOCAL:-/scratch}/${USER:?}/hydragnn-overlay-build-$$"
LOCAL_OVERLAY="${LOCAL_TMP}/hydragnn-overlay.img"
STAGE="${LOCAL_TMP}/stage"
PYG_WHEELS_DIR="${LOCAL_TMP}/pyg-rocm-wheels"

echo "=== HydraGNN overlay build ==="
echo "  SIF          : $HG_SIF"
echo "  Overlay dest : $OVERLAY  (${OVERLAY_SIZE_MB} MB)"
echo "  Build tmp    : $LOCAL_TMP"
echo "  HydraGNN SHA : $HG_HYDRAGNN_SHA"
echo "  PyG wheels   : Looong01 release ${PYG_ROCM_RELEASE} (torch 2.10, ROCm 7.1, ${PYG_PYTHON_TAG})"
echo "  Node         : $(hostname)  Date: $(date)"
echo ""

if [[ -f "$OVERLAY" ]]; then
  echo "WARNING: Overlay already exists: $OVERLAY"
  echo "  Delete it to rebuild: rm $OVERLAY"
  echo "  Proceeding with rebuild (will overwrite)..."
  rm -f "$OVERLAY"
fi

# ---------------------------------------------------------------------------
# Ensure MPI is available on this compute node
# ---------------------------------------------------------------------------
OMPI_DIR="/opt/ompi"
if [[ ! -x "${OMPI_DIR}/bin/mpicc" ]]; then
  echo "ERROR: /opt/ompi/bin/mpicc not found. Run this script on a compute node." >&2
  exit 1
fi
echo "  MPI: ${OMPI_DIR}/bin/mpicc ($(${OMPI_DIR}/bin/ompi_info --version 2>&1 | head -1))"
echo ""

mkdir -p "$LOCAL_TMP" "$STAGE" "$PYG_WHEELS_DIR"

# ---------------------------------------------------------------------------
# Download and extract Looong01 PyG ROCm wheels
# ---------------------------------------------------------------------------
PYG_ZIP="${LOCAL_TMP}/${PYG_ZIP_NAME}"
echo "--- Downloading PyG ROCm wheels ---"
curl -fL "$PYG_ZIP_URL" -o "$PYG_ZIP"
unzip -q -o "$PYG_ZIP" -d "$PYG_WHEELS_DIR"
echo "  Wheels: $(ls "$PYG_WHEELS_DIR"/*.whl 2>/dev/null | wc -l) files"

# ---------------------------------------------------------------------------
# Write the install script that runs inside the container
# ---------------------------------------------------------------------------
cat > "${LOCAL_TMP}/overlay_install.sh" << 'INNEREOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

PKG=/opt/hydragnn-pkgs
STAGE="$BUILD_STAGE"
PYG_WHEELS="$BUILD_PYG_WHEELS"
OMPI="/opt/ompi"

rm -rf "$STAGE" && mkdir -p "$STAGE"

echo "--- torch version: $(python3 -c 'import torch; print(torch.__version__)') ---"
echo "--- Installing to staging dir: $STAGE ---"

echo "--- Step 1: PyG ROCm wheels (torch-scatter, torch-sparse, torch-cluster) ---"
pip install -q --no-cache-dir --no-deps --target "$STAGE" "$PYG_WHEELS"/*.whl
echo "  installed: $(ls $STAGE | grep torch | tr '\n' ' ')"

echo "--- Step 2: torch-geometric (pure Python) ---"
pip install -q --no-cache-dir --target "$STAGE" torch-geometric 2>&1 | tail -3

echo "--- Step 3: HydraGNN runtime deps ---"
pip install -q --no-cache-dir --target "$STAGE" \
    mpi4py tqdm pyyaml tensorboard matplotlib \
    scikit-learn scipy h5py ase vesin e3nn 2>&1 | tail -5

echo "--- Step 4a: install build deps for adios2 (cmake, nanobind, mpi4py) ---"
export PATH="${OMPI}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMPI}/lib:${LD_LIBRARY_PATH:-}"
echo "  mpicc: $(which mpicc) ($(mpicc --showme:version 2>&1 | head -1))"
pip install -q --no-cache-dir cmake nanobind mpi4py 2>&1 | tail -3

echo "--- Step 4b: adios2 with MPI (cmake from source) ---"
pip download --no-deps --no-binary adios2 -d /tmp/adios2-dl adios2 2>&1 | tail -2
cd /tmp/adios2-dl && tar xf adios2*.tar.gz
ADIOS2_SRC=$(ls -d /tmp/adios2-dl/adios2-*/)
ADIOS2_BUILD=/tmp/adios2-build
ADIOS2_INST=/tmp/adios2-install
mkdir -p "$ADIOS2_BUILD"
cmake "$ADIOS2_SRC" -B "$ADIOS2_BUILD" \
    -DCMAKE_INSTALL_PREFIX="$ADIOS2_INST" \
    -DCMAKE_C_COMPILER="${OMPI}/bin/mpicc" \
    -DCMAKE_CXX_COMPILER="${OMPI}/bin/mpicxx" \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_Python=ON \
    -DADIOS2_USE_SST=OFF -DADIOS2_USE_OpenSSL=OFF -DADIOS2_USE_CURL=OFF \
    -DADIOS2_USE_Sodium=OFF -DADIOS2_USE_DAOS=OFF -DADIOS2_USE_IME=OFF \
    -DADIOS2_USE_MGARD=OFF -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Table=OFF \
    -DADIOS2_USE_Profiling=OFF -DBUILD_TESTING=OFF \
    2>&1 | grep -E "MPI|Python|nanobind|Configuring done"
cmake --build "$ADIOS2_BUILD" -j$(nproc) 2>&1 | tail -3
cmake --install "$ADIOS2_BUILD" 2>&1 | tail -3
# Copy Python package + libs to staging
cp -r "$ADIOS2_INST"/lib/python3.12/site-packages/adios2 "$STAGE/adios2"
cp "$ADIOS2_INST"/lib/libadios2*.so* "$STAGE/adios2/" 2>/dev/null || true
echo "  adios2 installed (MPI + Python bindings)"
cd /

echo "--- Step 5: HydraGNN source (main branch) ---"
if [[ ! -d /tmp/HydraGNN-src ]]; then
    git clone -q --depth=1 https://github.com/ORNL/HydraGNN.git /tmp/HydraGNN-src
    cd /tmp/HydraGNN-src && git fetch --depth=1 origin "$BUILD_SHA" && git checkout "$BUILD_SHA"
    cd /
fi
# Copy the source tree directly rather than pip-installing (avoids partial package issues)
cp -r /tmp/HydraGNN-src/hydragnn "$STAGE/hydragnn"
echo "  HydraGNN source copied (SHA: $BUILD_SHA)"

echo "--- Step 6: strip torch / nvidia / triton / cuda from staging dir ---"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton cuda sympy; do
    rm -rf "${STAGE}/${pkg}" "${STAGE}/${pkg}"-*.dist-info 2>/dev/null || true
done
rm -rf "${STAGE}"/nvidia_*.dist-info "${STAGE}"/cuda_*.dist-info 2>/dev/null || true
echo "  Staging size after strip: $(du -sh $STAGE | cut -f1)"

echo "--- Step 7: copy stripped packages → overlay ---"
mkdir -p "$PKG"
cp -r "$STAGE/." "$PKG/"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton cuda sympy; do
    rm -rf "${PKG}/${pkg}" "${PKG}/${pkg}"-*.dist-info 2>/dev/null || true
done
rm -rf "${PKG}"/nvidia_*.dist-info "${PKG}"/cuda_*.dist-info 2>/dev/null || true
# Keep adios2 .so libs accessible
echo "  Overlay contents: $(du -sh $PKG | cut -f1)"
echo "  adios2 libs: $(ls $PKG/adios2/libadios2_core_mpi.so 2>/dev/null && echo 'present' || echo 'MISSING')"

echo "--- Step 8: verify ---"
export PYTHONPATH="$PKG"
export LD_LIBRARY_PATH="$PKG/adios2:${LD_LIBRARY_PATH:-}"
python3 -c "
import torch, torch_scatter, torch_geometric
print('torch         :', torch.__version__)
print('torch_scatter : ok')
print('torch_geometric:', torch_geometric.__version__)
assert torch.cuda.is_available() or 'rocm' in torch.__version__, 'ROCm torch expected'
print('ROCm torch    : ok')
"

python3 -c "
import adios2
from mpi4py import MPI
a = adios2.Adios(MPI.COMM_SELF)
print('adios2        :', adios2.__version__, '(MPI: OK)')
"

python3 -c "import vesin; print('vesin         : ok')"
python3 -c "import e3nn; print('e3nn          :', e3nn.__version__)"
python3 -c "import hydragnn; print('hydragnn      : ok')"

echo "=== Overlay install complete ==="
INNEREOF

chmod +x "${LOCAL_TMP}/overlay_install.sh"

# ---------------------------------------------------------------------------
# Create overlay on local disk, run install, then copy to NFS
# ---------------------------------------------------------------------------
echo "Creating ${OVERLAY_SIZE_MB} MB ext3 overlay: $LOCAL_OVERLAY ..."
apptainer overlay create --size "$OVERLAY_SIZE_MB" "$LOCAL_OVERLAY"

echo "Running install inside container (binding /opt/ompi for MPI) ..."
apptainer exec \
    --rocm \
    --overlay "${LOCAL_OVERLAY}:rw" \
    --bind "${OMPI_DIR}:${OMPI_DIR}:ro" \
    --bind "${LOCAL_TMP}:${LOCAL_TMP}" \
    --env BUILD_STAGE="$STAGE" \
    --env BUILD_PYG_WHEELS="$PYG_WHEELS_DIR" \
    --env BUILD_SHA="$HG_HYDRAGNN_SHA" \
    "$HG_SIF" \
    bash "${LOCAL_TMP}/overlay_install.sh"

echo ""
echo "--- Copying finished overlay to shared storage: $OVERLAY ---"
mkdir -p "$(dirname "$OVERLAY")"
echo "  Image size: $(du -sh "$LOCAL_OVERLAY" | cut -f1)"
cp "$LOCAL_OVERLAY" "$OVERLAY"
echo "  Done."

# Cleanup local tmp
rm -rf "$LOCAL_TMP"

echo ""
echo "=== Overlay ready ==="
ls -lh "$OVERLAY"
echo ""
echo "SHA pinned: $HG_HYDRAGNN_SHA"
echo "To use:  export HG_OVERLAY=$OVERLAY"
