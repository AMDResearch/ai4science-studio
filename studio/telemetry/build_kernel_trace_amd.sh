#!/bin/bash
# Build the Omnistat kernel-tracing library (libomnistat_trace.so) ONCE on a
# compute node, where the rocprofiler-sdk cmake package + GPU headers live
# (/opt/rocm). Loaded into a training run via ROCP_TOOL_LIBRARIES when the
# telemetry sbatch is launched with OMNISTAT_KERNEL_TRACE=1, giving per-kernel
# dispatch counts + durations (per-epoch structure) in the telemetry panel.
#
# Run once (from studio/telemetry, so logs/ resolves):
#   PERF_TOOLS_DIR=... sbatch build_kernel_trace_amd.sh
# Output:    $PERF_TOOLS_DIR/omnistat-src/build-trace/libomnistat_trace.so
#            (PERF_TOOLS_DIR defaults to $AI4S_SHARED_DIR/perf-tools)
#SBATCH --job-name=kt-build
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/kt_build_%j.log
#SBATCH --error=logs/kt_build_%j.log
set -uo pipefail

PERF_TOOLS_DIR="${PERF_TOOLS_DIR:-${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR or PERF_TOOLS_DIR}/perf-tools}"
SRC="${PERF_TOOLS_DIR}/omnistat-src"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:${CMAKE_PREFIX_PATH:-}"
export PATH="/opt/rocm/bin:$PATH"

echo "=== node: $(hostname) ==="
ls -d /opt/rocm*/lib/cmake/rocprofiler-sdk 2>&1 | head

# The trace lib links libcurl. The runtime lib (libcurl.so.4) is present but the
# dev headers/symlink are not — stage libcurl4-openssl-dev into a scratch prefix
# (same apt-get download + dpkg-deb -x pattern as the demo GTK-lib staging).
CURLDEV="/scratch/$USER/kt_curldev_$$"
mkdir -p "$CURLDEV"; cd /tmp
apt-get download libcurl4-openssl-dev 2>&1 | grep -vE "^Get:|^Fetched|^Reading|^W:" || true
for d in /tmp/libcurl4-openssl-dev*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$CURLDEV/"; done
_CURL_INC="$CURLDEV/usr/include/x86_64-linux-gnu:$CURLDEV/usr/include"
_CURL_LIBDIR="/usr/lib/x86_64-linux-gnu"   # runtime libcurl.so.4 lives here
# The dev .deb ships a libcurl.so -> libcurl.so.4 symlink; expose it too.
export CMAKE_INCLUDE_PATH="$_CURL_INC:${CMAKE_INCLUDE_PATH:-}"
export CMAKE_LIBRARY_PATH="$CURLDEV/usr/lib/x86_64-linux-gnu:$_CURL_LIBDIR:${CMAKE_LIBRARY_PATH:-}"

echo "=== configure (BUILD_KERNEL_TRACE_LIB=ON) ==="
rm -rf "$SRC/build-trace"
cmake -S "$SRC/rocprofiler-sdk" -B "$SRC/build-trace" \
  -DBUILD_KERNEL_TRACE_LIB=ON \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake" \
  -DCURL_INCLUDE_DIR="$CURLDEV/usr/include/x86_64-linux-gnu" \
  -DCURL_LIBRARY="$_CURL_LIBDIR/libcurl.so.4" 2>&1 | tail -20
echo "=== build ==="
cmake --build "$SRC/build-trace" -j 8 2>&1 | tail -20

SO=$(find "$SRC/build-trace" -name "libomnistat_trace.so" 2>/dev/null | head -1)
echo "built .so: $SO"
if [[ -n "$SO" && "$SO" != "$SRC/build-trace/libomnistat_trace.so" ]]; then
  cp -v "$SO" "$SRC/build-trace/libomnistat_trace.so"
fi
rm -rf "$CURLDEV" 2>/dev/null || true
[[ -f "$SRC/build-trace/libomnistat_trace.so" ]] && { echo "OK: $SRC/build-trace/libomnistat_trace.so"; ls -la "$SRC/build-trace/libomnistat_trace.so"; } \
  || { echo "ERROR: libomnistat_trace.so not built" >&2; exit 3; }
echo "=== END KT BUILD ==="
