#!/bin/bash
# Build + install the omnistat rocprofiler-sdk C-extension the same way the
# known-good reference tree did it: BUILD_ROCPROFILER_SDK_EXTENSION=1 pip install (which drives
# cmake_build_extension from setup.py), ON A COMPUTE NODE where the rocprofiler-sdk
# cmake package + GPU headers live (/opt/rocm). This produces a stripped release
# module installed into the venv's omnistat/ package, enabling FP64 + HBM counters.
#SBATCH --job-name=roc-ext
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=logs/roc_ext_%j.log
#SBATCH --error=logs/roc_ext_%j.log
set -uo pipefail

PERF_TOOLS_DIR="${PERF_TOOLS_DIR:-${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR or PERF_TOOLS_DIR}/perf-tools}"
VENV="${PERF_TOOLS_DIR}/omnistat-venv"
SRC="${PERF_TOOLS_DIR}/omnistat-src"
PY=$VENV/bin/python

echo "=== node: $(hostname) ==="
ls -d /opt/rocm*/lib/cmake/rocprofiler-sdk 2>&1 | head
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:${CMAKE_PREFIX_PATH:-}"
export PATH="/opt/rocm/bin:$PATH"
export BUILD_ROCPROFILER_SDK_EXTENSION=1

echo "=== pip install (drives setup.py cmake_build_extension) ==="
cd "$SRC"
# Remove any prior manual-cmake artifact so pip's build is authoritative.
rm -f "$VENV/lib/python3.12/site-packages/omnistat/rocprofiler_sdk_extension"*.so
"$PY" -m pip install --force-reinstall --no-deps ".[query]" 2>&1 | tail -30

echo "=== Verify extension present + stripped ==="
SO=$(find "$VENV/lib/python3.12/site-packages/omnistat" -name "rocprofiler_sdk_extension*.so" 2>/dev/null | head -1)
echo "installed .so: $SO"
[[ -n "$SO" ]] && { ls -la "$SO"; file "$SO"; } || { echo "ERROR: extension not installed" >&2; exit 3; }

echo "=== Import check ==="
$PY -c "from omnistat.rocprofiler_sdk_extension import get_samplers, initialize; print('import OK')" 2>&1 | head
echo "=== END BUILD ==="
