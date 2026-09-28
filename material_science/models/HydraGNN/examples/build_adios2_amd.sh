#!/usr/bin/env bash
# Build MPI-enabled ADIOS2 Python bindings against the cluster MPI stack.
#
# ORNL Lux: in-container cmake with the system OpenMPI module (software tree discovered at runtime).
# Vultr Lux: in-container build against /opt/ompi.
#
# Installs to a persistent prefix on shared storage; reuse from build_overlay_amd.sh.
#
# Quick start (ORNL Lux compute node):
#   export AI4S_SHARED_DIR=/path/to/shared
#   sbatch sbatch_build_adios2_amd.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lux_mpi_env.sh
source "${SCRIPT_DIR}/lux_mpi_env.sh"

HG_BASE="${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}/models/HydraGNN"
HG_SIF="${HG_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
HG_ADIOS2_PREFIX="${HG_ADIOS2_PREFIX:-${HG_BASE}/adios2-lux}"
LOCAL_TMP="${SCRATCH_LOCAL:-/tmp}/${USER:?}/hydragnn-adios2-build-${SLURM_JOB_ID:-$$}"
SRC_DIR="${LOCAL_TMP}/src"
PIP_DEPS_DIR="${LOCAL_TMP}/pip-deps"
CMAKE_DIR="${LOCAL_TMP}/cmake-build"

echo "=== HydraGNN ADIOS2 build ==="
echo "  SIF            : $HG_SIF"
echo "  Install prefix : $HG_ADIOS2_PREFIX"
echo "  Build tmp      : $LOCAL_TMP"
echo "  Node           : $(hostname)  Date: $(date)"
echo ""

MPI_BINDS=()
BUILD_OMPI_DIR=/opt/ompi
if lux_mpi_should_use && lux_mpi_setup_binds; then
  echo "  MPI baseline   : system ${HG_LUX_OPENMPI_MODULE}"
  BUILD_OMPI_DIR="${HG_OMPI_DIR}"
  export HG_MPI_EXTRA_LDPATH MPI4PY_LIBMPI HG_LUX_OPAL_PREFIX
elif [[ -x /opt/ompi/bin/mpicc ]]; then
  MPI_BINDS=( --bind /opt/ompi:/opt/ompi:ro )
  echo "  MPI baseline   : /opt/ompi"
else
  echo "ERROR: No MPI stack found. On ORNL Lux load the system OpenMPI module." >&2
  exit 1
fi
echo ""

mkdir -p "$LOCAL_TMP" "$SRC_DIR" "$PIP_DEPS_DIR" "$CMAKE_DIR" "$HG_ADIOS2_PREFIX"

cat > "${LOCAL_TMP}/mpi-verified.cmake" << EOF
set(MPI_C_COMPILER "${BUILD_OMPI_DIR}/bin/mpicc" CACHE FILEPATH "" FORCE)
set(MPI_CXX_COMPILER "${BUILD_OMPI_DIR}/bin/mpicxx" CACHE FILEPATH "" FORCE)
set(MPI_C_COMPILER_INCLUDE_DIRS "${BUILD_OMPI_DIR}/include" CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER_INCLUDE_DIRS "${BUILD_OMPI_DIR}/include" CACHE STRING "" FORCE)
set(MPI_C_COMPILER_LIBRARIES "${BUILD_OMPI_DIR}/lib/libmpi.so" CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER_LIBRARIES "${BUILD_OMPI_DIR}/lib/libmpi.so" CACHE STRING "" FORCE)
set(MPI_C_COMPILER_WORKS TRUE CACHE BOOL "" FORCE)
set(MPI_CXX_COMPILER_WORKS TRUE CACHE BOOL "" FORCE)
set(MPI_C_VERSION "5.0.10" CACHE STRING "" FORCE)
set(MPI_CXX_VERSION "5.0.10" CACHE STRING "" FORCE)
set(MPI_C_FOUND TRUE CACHE BOOL "" FORCE)
set(MPI_CXX_FOUND TRUE CACHE BOOL "" FORCE)
set(MPI_C_WORKS TRUE CACHE BOOL "" FORCE)
set(MPI_CXX_WORKS TRUE CACHE BOOL "" FORCE)
set(MPI_DETERMINE_LIBRARY_VERSION FALSE CACHE BOOL "" FORCE)
set(MPI_FOUND TRUE CACHE BOOL "" FORCE)
EOF

cat > "${LOCAL_TMP}/adios2_build.sh" << 'INNEREOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

OMPI="${BUILD_OMPI_DIR:-/opt/ompi}"
PIP_DEPS="${BUILD_PIP_DEPS}"
CMAKE_BUILD="${BUILD_CMAKE_DIR}"
mkdir -p "$PIP_DEPS" "$CMAKE_BUILD"

export PATH="${OMPI}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMPI}/lib:${HG_MPI_EXTRA_LDPATH:-}:${LD_LIBRARY_PATH:-}"
[[ -n "${HG_LUX_OPAL_PREFIX:-}" ]] && export OPAL_PREFIX="${HG_LUX_OPAL_PREFIX}"
[[ -n "${MPI4PY_LIBMPI:-}" ]] && export MPI4PY_LIBMPI="${MPI4PY_LIBMPI}"
export OMPI_CC="${OMPI_CC:-/usr/bin/gcc}"
export OMPI_CXX="${OMPI_CXX:-/usr/bin/g++}"

export PATH="${PIP_DEPS}/bin:${PATH}"
export PYTHONPATH="${PIP_DEPS}:${PYTHONPATH:-}"
pip install -q --no-cache-dir --target "$PIP_DEPS" cmake nanobind mpi4py

echo "  mpicc: ${OMPI}/bin/mpicc ($(${OMPI}/bin/mpicc --version 2>&1 | head -1))"

_LINK_FLAGS="-L${OMPI}/lib -Wl,-rpath,${OMPI}/lib"
if [[ -n "${HG_MPI_EXTRA_LDPATH:-}" ]]; then
  IFS=':' read -ra _ld <<< "${HG_MPI_EXTRA_LDPATH}"
  for _p in "${_ld[@]}"; do
    [[ -d "$_p" ]] && _LINK_FLAGS+=" -L${_p} -Wl,-rpath,${_p}"
  done
fi

pip download --no-deps --no-binary adios2 -d "$BUILD_SRC" adios2
cd "$BUILD_SRC" && tar xf adios2*.tar.gz
ADIOS2_SRC=$(ls -d adios2-*/)

cmake -C "${BUILD_MPI_CACHE}" \
  -S "$ADIOS2_SRC" -B "$CMAKE_BUILD" \
  -DCMAKE_INSTALL_PREFIX="$HG_ADIOS2_PREFIX" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DMPI_C_COMPILER="${OMPI}/bin/mpicc" \
  -DMPI_CXX_COMPILER="${OMPI}/bin/mpicxx" \
  -DCMAKE_EXE_LINKER_FLAGS="${_LINK_FLAGS}" \
  -DCMAKE_SHARED_LINKER_FLAGS="${_LINK_FLAGS}" \
  -DADIOS2_USE_MPI=ON \
  -DADIOS2_USE_Python=ON \
  -DADIOS2_USE_SST=OFF \
  -DADIOS2_USE_OpenSSL=OFF \
  -DADIOS2_USE_CURL=OFF \
  -DADIOS2_USE_Sodium=OFF \
  -DADIOS2_USE_DAOS=OFF \
  -DADIOS2_USE_IME=OFF \
  -DADIOS2_USE_MGARD=OFF \
  -DADIOS2_USE_Fortran=OFF \
  -DADIOS2_USE_Table=OFF \
  -DADIOS2_USE_Profiling=OFF \
  -DADIOS2_BUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF

set +e
cmake --build "$CMAKE_BUILD" -j"$(nproc)"
set -e
cmake --build "$CMAKE_BUILD" -j"$(nproc)" --target adios2_py
_bindings_so=$(find "$CMAKE_BUILD/lib/python3.12/site-packages/adios2/bindings" -name '*.so' -print -quit 2>/dev/null || true)
if [[ ! -f "$CMAKE_BUILD/lib/libadios2_core_mpi.so" ]]; then
  echo "ERROR: libadios2_core_mpi.so was not built" >&2
  exit 1
fi
if [[ -z "$_bindings_so" || ! -f "$_bindings_so" ]]; then
  echo "ERROR: adios2 Python bindings were not built" >&2
  exit 1
fi
echo "  core MPI + Python bindings: built (optional utility binaries may be skipped on Lux)"

set +e
cmake --install "$CMAKE_BUILD"
set -e
if [[ ! -f "$HG_ADIOS2_PREFIX/lib/libadios2_core_mpi.so" ]]; then
  echo "ERROR: install missing libadios2_core_mpi.so" >&2
  exit 1
fi
if [[ ! -d "$HG_ADIOS2_PREFIX/lib/python3.12/site-packages/adios2" ]]; then
  echo "  completing Python bindings install from build tree..."
  mkdir -p "$HG_ADIOS2_PREFIX/lib/python3.12/site-packages"
  cp -a "$CMAKE_BUILD/lib/python3.12/site-packages/adios2" "$HG_ADIOS2_PREFIX/lib/python3.12/site-packages/"
fi
if [[ -d "$CMAKE_BUILD/lib/python3.12/site-packages/adios2" ]]; then
  cp -a "$CMAKE_BUILD/lib/libadios2"*.so* "$HG_ADIOS2_PREFIX/lib/" 2>/dev/null || true
fi

python3 -c "
import sys
sys.path.insert(0, '${PIP_DEPS}')
sys.path.insert(0, '${HG_ADIOS2_PREFIX}/lib/python3.12/site-packages')
import adios2
from mpi4py import MPI
adios2.Adios(MPI.COMM_SELF)
print('adios2 verify: OK', adios2.__version__)
"
INNEREOF
chmod +x "${LOCAL_TMP}/adios2_build.sh"

PROXY_ENVS=()
for k in https_proxy http_proxy HTTPS_PROXY HTTP_PROXY no_proxy NO_PROXY; do
  v="${!k:-}"
  [[ -n "$v" ]] && PROXY_ENVS+=( --env "${k}=${v}" )
done

BUILD_ENVS=(
  --env BUILD_OMPI_DIR="$BUILD_OMPI_DIR"
  --env BUILD_SRC="$SRC_DIR"
  --env BUILD_PIP_DEPS="$PIP_DEPS_DIR"
  --env BUILD_CMAKE_DIR="$CMAKE_DIR"
  --env BUILD_MPI_CACHE="${LOCAL_TMP}/mpi-verified.cmake"
  --env HG_ADIOS2_PREFIX="$HG_ADIOS2_PREFIX"
)
[[ -n "${HG_LUX_OPAL_PREFIX:-}" ]] && BUILD_ENVS+=( --env HG_LUX_OPAL_PREFIX="$HG_LUX_OPAL_PREFIX" )
[[ -n "${MPI4PY_LIBMPI:-}" ]] && BUILD_ENVS+=( --env MPI4PY_LIBMPI="$MPI4PY_LIBMPI" )
[[ -n "${HG_MPI_EXTRA_LDPATH:-}" ]] && BUILD_ENVS+=( --env HG_MPI_EXTRA_LDPATH="$HG_MPI_EXTRA_LDPATH" )

unset LD_PRELOAD
apptainer exec --rocm \
  "${MPI_BINDS[@]}" \
  --bind "${LOCAL_TMP}:${LOCAL_TMP}" \
  --bind "${HG_ADIOS2_PREFIX}:${HG_ADIOS2_PREFIX}" \
  "${PROXY_ENVS[@]}" \
  "${BUILD_ENVS[@]}" \
  "$HG_SIF" \
  bash "${LOCAL_TMP}/adios2_build.sh"

rm -rf "$LOCAL_TMP"

echo ""
echo "=== ADIOS2 ready ==="
echo "  Prefix : $HG_ADIOS2_PREFIX"
echo "  Export : export HG_ADIOS2_PREFIX=$HG_ADIOS2_PREFIX"
