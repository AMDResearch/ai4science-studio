#!/usr/bin/env bash
# Build a persistent Apptainer ext3 overlay pre-loaded with all HydraGNN pip
# dependencies.  Run once per cluster; reuse the overlay image across jobs to
# skip the pip install phase on every submission.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   # ORNL Lux (two steps on a compute node):
#   sbatch sbatch_build_adios2_amd.sh    # once — MPI ADIOS2 to shared prefix
#   sbatch sbatch_build_overlay_amd.sh   # PyG wheels + hydragnn into overlay
#
#   # Vultr Lux (/opt/ompi):
#   bash build_adios2_amd.sh && bash build_overlay_amd.sh
#
# ── MPI-enabled adios2 ──────────────────────────────────────────────────────
#   Built separately by build_adios2_amd.sh (standard cmake install).
#   ORNL Lux uses the site openmpi/5.0.10 module via lux_mpi_env.sh.
#
# ── torch-scatter / torch-sparse / torch-cluster / torch-spline-conv ────────
#   No official ROCm wheels exist at data.pyg.org.  We use pre-built ROCm
#   wheels from https://github.com/Looong01/pyg-rocm-build (release 15).
#
# ── Key environment variables ────────────────────────────────────────────────
#   AI4S_SHARED_DIR     Shared filesystem root (required)
#   HG_ADIOS2_PREFIX    ADIOS2 install tree from build_adios2_amd.sh
#   HG_SIF              Path to the Apptainer SIF image
#   HG_OVERLAY          Output overlay path
#   HG_OVERLAY_SIZE_MB  Size of the ext3 overlay in MB (default: 4096)
#   HG_HYDRAGNN_SHA     Git SHA to pin HydraGNN at

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lux_mpi_env.sh
source "${SCRIPT_DIR}/lux_mpi_env.sh"

HG_BASE="${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}/models/HydraGNN"
HG_SIF="${HG_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
HG_ADIOS2_PREFIX="${HG_ADIOS2_PREFIX:-${HG_BASE}/adios2-lux}"
OVERLAY="${HG_OVERLAY:-${HG_BASE}/overlays/hydragnn-overlay.img}"
OVERLAY_SIZE_MB="${HG_OVERLAY_SIZE_MB:-4096}"
HG_HYDRAGNN_SHA="${HG_HYDRAGNN_SHA:-2fb0bd0157e3c85a74f9841887155095bd163303}"
PYG_ROCM_RELEASE="${PYG_ROCM_RELEASE:-15}"
PYG_PYTHON_TAG="${PYG_PYTHON_TAG:-py312}"

PYG_ZIP_NAME="torch-2.10.0-rocm-7.1-${PYG_PYTHON_TAG}-linux_x86_64.zip"
PYG_ZIP_URL="https://github.com/Looong01/pyg-rocm-build/releases/download/${PYG_ROCM_RELEASE}/${PYG_ZIP_NAME}"

LOCAL_TMP="${SCRATCH_LOCAL:-/tmp}/${USER:?}/hydragnn-overlay-build-$$"
LOCAL_OVERLAY="${LOCAL_TMP}/hydragnn-overlay.img"
STAGE="${LOCAL_TMP}/stage"
PYG_WHEELS_DIR="${LOCAL_TMP}/pyg-rocm-wheels"

echo "=== HydraGNN overlay build ==="
echo "  SIF            : $HG_SIF"
echo "  ADIOS2 prefix  : $HG_ADIOS2_PREFIX"
echo "  Overlay dest   : $OVERLAY  (${OVERLAY_SIZE_MB} MB)"
echo "  Build tmp      : $LOCAL_TMP"
echo "  HydraGNN SHA   : $HG_HYDRAGNN_SHA"
echo "  Node           : $(hostname)  Date: $(date)"
echo ""

if [[ ! -f "${HG_ADIOS2_PREFIX}/lib/libadios2_core_mpi.so" ]]; then
  echo "ERROR: MPI ADIOS2 not found at $HG_ADIOS2_PREFIX" >&2
  echo "  Run first: sbatch sbatch_build_adios2_amd.sh  (ORNL Lux compute node)" >&2
  exit 1
fi

if [[ -f "$OVERLAY" ]]; then
  echo "WARNING: Overlay already exists — overwriting: $OVERLAY"
  rm -f "$OVERLAY"
fi

# MPI binds needed for overlay verify (adios2 loads host MPI at import time).
MPI_BINDS=()
if lux_mpi_should_use && lux_mpi_setup_binds; then
  echo "  MPI runtime  : system ${HG_LUX_OPENMPI_MODULE}"
elif [[ -x /opt/ompi/bin/mpicc ]]; then
  MPI_BINDS=( --bind /opt/ompi:/opt/ompi:ro )
  echo "  MPI runtime  : /opt/ompi"
fi
echo ""

mkdir -p "$LOCAL_TMP" "$STAGE" "$PYG_WHEELS_DIR"

PATCHES_SRC="${SCRIPT_DIR}/patches"
PATCHES_STAGE="${LOCAL_TMP}/patches"
mkdir -p "$PATCHES_STAGE"
if [[ -d "$PATCHES_SRC" ]]; then
  shopt -s nullglob
  for p in "$PATCHES_SRC"/*.patch; do cp "$p" "$PATCHES_STAGE/"; done
  shopt -u nullglob
fi

PYG_ZIP="${LOCAL_TMP}/${PYG_ZIP_NAME}"
echo "--- Downloading PyG ROCm wheels ---"
curl -fL "$PYG_ZIP_URL" -o "$PYG_ZIP"
unzip -q -o "$PYG_ZIP" -d "$PYG_WHEELS_DIR"
echo "  Wheels: $(ls "$PYG_WHEELS_DIR"/*.whl 2>/dev/null | wc -l) files"

cat > "${LOCAL_TMP}/overlay_install.sh" << 'INNEREOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

PKG=/opt/hydragnn-pkgs
STAGE="$BUILD_STAGE"
PYG_WHEELS="$BUILD_PYG_WHEELS"
OMPI="${BUILD_OMPI_DIR:-}"
if [[ -n "$OMPI" ]]; then
  export PATH="${OMPI}/bin:${PATH}"
  export LD_LIBRARY_PATH="${OMPI}/lib:${HG_MPI_EXTRA_LDPATH:-}:${LD_LIBRARY_PATH:-}"
  [[ -n "${HG_LUX_OPAL_PREFIX:-}" ]] && export OPAL_PREFIX="${HG_LUX_OPAL_PREFIX}"
  [[ -n "${MPI4PY_LIBMPI:-}" ]] && export MPI4PY_LIBMPI
fi

rm -rf "$STAGE" && mkdir -p "$STAGE"

echo "--- Step 1: PyG ROCm wheels ---"
pip install -q --no-cache-dir --no-deps --target "$STAGE" "$PYG_WHEELS"/*.whl

echo "--- Step 2: torch-geometric ---"
pip install -q --no-cache-dir --target "$STAGE" torch-geometric

echo "--- Step 3: HydraGNN runtime deps ---"
pip install -q --no-cache-dir --target "$STAGE" \
    mpi4py tqdm pyyaml tensorboard matplotlib \
    scikit-learn scipy h5py ase vesin e3nn

echo "--- Step 4: ADIOS2 from install prefix ---"
cp -r "$BUILD_ADIOS2_PREFIX/lib/python3.12/site-packages/adios2" "$STAGE/adios2"
cp "$BUILD_ADIOS2_PREFIX/lib/libadios2"*.so* "$STAGE/adios2/"
echo "  adios2 libs: $(ls "$STAGE/adios2"/libadios2_core_mpi.so && echo OK)"

echo "--- Step 5: HydraGNN source (SHA: $BUILD_SHA) ---"
git clone -q https://github.com/ORNL/HydraGNN.git /tmp/HydraGNN-src
cd /tmp/HydraGNN-src && git checkout -q "$BUILD_SHA"
if [[ -d "$BUILD_PATCHES" ]]; then
  for p in "$BUILD_PATCHES"/*.patch; do
    [[ -f "$p" ]] || continue
    patch -p1 --dry-run --silent < "$p"
    patch -p1 < "$p"
  done
fi
cp -r /tmp/HydraGNN-src/hydragnn "$STAGE/hydragnn"

echo "--- Step 6: strip torch from staging ---"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton cuda sympy; do
  rm -rf "${STAGE}/${pkg}" "${STAGE}/${pkg}"-*.dist-info 2>/dev/null || true
done

echo "--- Step 7: copy → overlay ---"
mkdir -p "$PKG"
cp -r "$STAGE/." "$PKG/"
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton cuda sympy; do
  rm -rf "${PKG}/${pkg}" "${PKG}/${pkg}"-*.dist-info 2>/dev/null || true
done

echo "--- Step 8: verify ---"
export PYTHONPATH="$PKG"
export LD_LIBRARY_PATH="$PKG/adios2:${LD_LIBRARY_PATH:-}"
python3 -c "import torch; assert torch.cuda.is_available() or 'rocm' in torch.__version__"
python3 -c "import torch_scatter, torch_geometric, adios2; from mpi4py import MPI; adios2.Adios(MPI.COMM_SELF)"
python3 -c "import hydragnn, vesin, e3nn"
echo "=== Overlay install complete ==="
INNEREOF
chmod +x "${LOCAL_TMP}/overlay_install.sh"

echo "Creating ${OVERLAY_SIZE_MB} MB overlay: $LOCAL_OVERLAY ..."
apptainer overlay create --size "$OVERLAY_SIZE_MB" "$LOCAL_OVERLAY"

PROXY_ENVS=()
for k in https_proxy http_proxy HTTPS_PROXY HTTP_PROXY no_proxy NO_PROXY; do
  v="${!k:-}"
  [[ -n "$v" ]] && PROXY_ENVS+=( --env "${k}=${v}" )
done

BUILD_ENVS=(
  --env BUILD_STAGE="$STAGE"
  --env BUILD_PYG_WHEELS="$PYG_WHEELS_DIR"
  --env BUILD_SHA="$HG_HYDRAGNN_SHA"
  --env BUILD_PATCHES="$PATCHES_STAGE"
  --env BUILD_ADIOS2_PREFIX="$HG_ADIOS2_PREFIX"
)
[[ -n "${HG_OMPI_DIR:-}" ]] && BUILD_ENVS+=( --env BUILD_OMPI_DIR="$HG_OMPI_DIR" )
[[ -n "${HG_LUX_OPAL_PREFIX:-}" ]] && BUILD_ENVS+=( --env HG_LUX_OPAL_PREFIX="$HG_LUX_OPAL_PREFIX" )
[[ -n "${MPI4PY_LIBMPI:-}" ]] && BUILD_ENVS+=( --env MPI4PY_LIBMPI="$MPI4PY_LIBMPI" )
[[ -n "${HG_MPI_EXTRA_LDPATH:-}" ]] && BUILD_ENVS+=( --env HG_MPI_EXTRA_LDPATH="$HG_MPI_EXTRA_LDPATH" )

unset LD_PRELOAD
apptainer exec --rocm \
  --overlay "${LOCAL_OVERLAY}:rw" \
  "${MPI_BINDS[@]}" \
  --bind "${LOCAL_TMP}:${LOCAL_TMP}" \
  --bind "${HG_ADIOS2_PREFIX}:${HG_ADIOS2_PREFIX}:ro" \
  "${PROXY_ENVS[@]}" \
  "${BUILD_ENVS[@]}" \
  "$HG_SIF" \
  bash "${LOCAL_TMP}/overlay_install.sh"

mkdir -p "$(dirname "$OVERLAY")"
cp "$LOCAL_OVERLAY" "$OVERLAY"
rm -rf "$LOCAL_TMP"

echo ""
echo "=== Overlay ready ==="
ls -lh "$OVERLAY"
