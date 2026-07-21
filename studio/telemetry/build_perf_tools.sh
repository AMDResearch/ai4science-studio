#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# build_perf_tools.sh
#
# Build a fresh, SELF-OWNED AMD Omnistat performance-monitoring venv on the lux
# MI355X cluster login node (rad-vultr-login) under /shared/spannala, so we no
# longer depend on a colleague's non-relocatable copy at
# /shared/omnihub/tools/omnistat/venv (whose shebangs are hardcoded to
# /shared/omnihub and therefore cannot be moved/reused).
#
# WHAT IT PRODUCES
#   /shared/spannala/perf-tools/omnistat-venv        -- the venv
#   /shared/spannala/perf-tools/omnistat-src         -- omnistat source checkout
#   .../omnistat-venv/bin/omnistat-usermode          -- launcher (pure Python)
#
# ROCPROFILER-SDK C-EXTENSION (REQUIRED for FP64/HBM counters)
#   Omnistat's pyproject builds an optional nanobind/rocprofiler-sdk C extension
#   only when BUILD_ROCPROFILER_SDK_EXTENSION=1 (see setup.py). This step here is
#   a PURE-PYTHON install (extension NOT built) because the extension needs the
#   rocprofiler-sdk cmake package + GPU headers that exist ONLY on a compute node,
#   not this login node. WITHOUT the extension, the exporter ABORTS at startup
#   ("Missing ROCProfiler-SDK extension") whenever enable_rocprofiler=True — so
#   telemetry silently fails. Build it separately on a compute node via SLURM:
#       sbatch studio/telemetry/build_rocprofiler_ext.sbatch
#   which compiles the .so against /opt/rocm and drops it into this venv's
#   omnistat/ package. See SKILL lesson #33.
#
# IDEMPOTENT: if the venv already exists and omnistat-usermode --help works,
# the script exits early without rebuilding.
#
# Run ON the login node (it has network access; compute nodes do not).
# -----------------------------------------------------------------------------
set -euo pipefail

PERF_ROOT="/shared/spannala/perf-tools"
VENV="${PERF_ROOT}/omnistat-venv"
SRC="${PERF_ROOT}/omnistat-src"
USERMODE="${VENV}/bin/omnistat-usermode"

# Reference source tree owned by the colleague (READ-ONLY; used only to pin the
# exact commit so our build matches what is known-good on this cluster).
REF_SRC="/shared/omnihub/tools/omnistat-src"
OMNISTAT_REPO="https://github.com/ROCm/omnistat.git"

# --- idempotency check -------------------------------------------------------
if [[ -x "${USERMODE}" ]] && "${USERMODE}" --help >/dev/null 2>&1; then
    echo "[build_perf_tools] omnistat-usermode already works at ${USERMODE}; nothing to do."
    exit 0
fi

# --- pick a python (>=3.9) ---------------------------------------------------
PY="$(command -v python3)"
PYVER="$("${PY}" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "[build_perf_tools] using ${PY} (Python ${PYVER})"
"${PY}" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)' \
    || { echo "ERROR: need Python >= 3.9, found ${PYVER}" >&2; exit 1; }

mkdir -p "${PERF_ROOT}"

# --- determine the commit to build ------------------------------------------
# Match the commit aaji's known-good tree is on, if we can read it.
PIN_COMMIT=""
if git -C "${REF_SRC}" rev-parse HEAD >/dev/null 2>&1; then
    PIN_COMMIT="$(git -C "${REF_SRC}" rev-parse HEAD)"
else
    # 'dubious ownership' guard: register the read-only tree as safe, then retry.
    git config --global --add safe.directory "${REF_SRC}" >/dev/null 2>&1 || true
    PIN_COMMIT="$(git -C "${REF_SRC}" rev-parse HEAD 2>/dev/null || true)"
fi
echo "[build_perf_tools] target commit: ${PIN_COMMIT:-<unknown, will use repo default>}"

# --- obtain source ----------------------------------------------------------
# NOTE: the reference tree (aaji's) sits on a LOCAL commit (v1.12.0-18-g65ea9ac)
# that was never pushed to the public ROCm/omnistat remote, so a fresh clone
# CANNOT reproduce that exact commit. To honor "same commit as aaji's tree" we
# therefore COPY the reference tree (stripping its build artifacts). If the
# reference tree is unreadable, we fall back to a fresh public clone.
copy_reference_tree() {
    rm -rf "${SRC}"
    cp -r "${REF_SRC}" "${SRC}"
    chmod -R u+w "${SRC}"
    # Strip the colleague's build artifacts so it rebuilds clean.
    rm -rf "${SRC}/build" "${SRC}/build-trace" "${SRC}"/*.egg-info \
           "${SRC}/omnistat.egg-info"
    echo "[build_perf_tools] copied reference tree into ${SRC} at $(git -C "${SRC}" rev-parse HEAD 2>/dev/null || echo '?')"
}

fresh_clone() {
    rm -rf "${SRC}"
    git clone "${OMNISTAT_REPO}" "${SRC}"
    if [[ -n "${PIN_COMMIT}" ]]; then
        git -C "${SRC}" fetch --tags origin "${PIN_COMMIT}" 2>/dev/null || true
        git -C "${SRC}" checkout -q "${PIN_COMMIT}" 2>/dev/null \
            || echo "[build_perf_tools] WARN: pinned commit ${PIN_COMMIT} not on public remote; using clone default $(git -C "${SRC}" rev-parse HEAD)." >&2
    fi
    echo "[build_perf_tools] cloned ${OMNISTAT_REPO} at $(git -C "${SRC}" rev-parse HEAD)"
}

get_source() {
    # Prefer an exact-commit copy of the known-good reference tree.
    if git -C "${REF_SRC}" cat-file -t "${PIN_COMMIT}" >/dev/null 2>&1; then
        copy_reference_tree && return 0
    fi
    # Otherwise try a fresh public clone; if THAT fails, copy as last resort.
    if fresh_clone; then return 0; fi
    echo "[build_perf_tools] clone failed; copying reference tree as fallback." >&2
    copy_reference_tree
}
get_source

# --- create the venv ---------------------------------------------------------
if [[ ! -x "${VENV}/bin/python" ]]; then
    "${PY}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# --- install omnistat (pure Python; C extension intentionally NOT built) -----
# Extras group [query] pulls numpy/pandas/matplotlib/etc for omnistat-query.
export BUILD_ROCPROFILER_SDK_EXTENSION=0
cd "${SRC}"
if ! python -m pip install ".[query]"; then
    echo "[build_perf_tools] '.[query]' failed; retrying core-only install." >&2
    python -m pip install .
fi

deactivate

# --- patch upstream exporter-availability bug --------------------------------
# omni_util.py's check_exporter() has a `return False` mis-indented INSIDE the
# retry for-loop, so it gives up after ONE attempt. When rocprofiler init takes
# >5s the exporter check intermittently reports "0 of N exporters available"
# even though the exporter is fine. Dedent that return so the 24-iteration
# backoff (~15s) actually runs. Idempotent (grep guard). See SKILL lesson #33.
OMNI_UTIL="${VENV}/lib/python${PYVER}/site-packages/omnistat/omni_util.py"
if [[ -f "${OMNI_UTIL}" ]] && ! grep -q "studio/telemetry.*dedent" "${OMNI_UTIL}"; then
    "${VENV}/bin/python" - "${OMNI_UTIL}" <<'PYPATCH'
import re, sys
p = sys.argv[1]
s = open(p).read()
# The buggy block: `time.sleep(delay)` ... except ... `return False` (16-sp indent)
# ending the for-body. Replace the final 16-space `return False` that immediately
# follows the except-block with a dedented (12-space) one carrying a marker.
buggy = ("                    except Exception:\n"
         "                        return False\n"
         "                return False\n")
fixed = ("                    except Exception:\n"
         "                        return False\n"
         "            # studio/telemetry dedent fix: retry loop must exhaust\n"
         "            return False\n")
if buggy in s:
    s = s.replace(buggy, fixed)
    open(p, "w").write(s)
    print("[build_perf_tools] patched exporter retry loop in omni_util.py")
else:
    print("[build_perf_tools] exporter retry block not found (upstream changed?) — skipping")
PYPATCH
fi

# --- verify ------------------------------------------------------------------
echo "[build_perf_tools] verifying ..."
"${USERMODE}" --help | head -15
echo "[build_perf_tools] shebang: $(head -1 "${USERMODE}")"
echo "[build_perf_tools] DONE. Self-owned Omnistat venv ready at ${VENV}"
