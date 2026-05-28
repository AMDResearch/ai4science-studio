#!/usr/bin/env bash
# Compare /shared/omnihub Omnistat installs vs ai4science omnistat-pr271 workflow.
# Run on Lux login node. Writes report to stdout; optional --output file.
set -euo pipefail

SHARED_OMNI="${OMNIHUB_SHARED_DIR:-/shared/omnihub}"
AI4S_SHARED="${AI4S_SHARED_DIR:-/shared/$USER}"
OUT="${1:-}"

report() {
  if [[ -n "$OUT" ]]; then
    tee -a "$OUT"
  else
    cat
  fi
}

{
  echo "# Omnistat parity check — $(date -Iseconds)"
  echo ""
  echo "## /shared/omnihub/tools/omnistat"
  echo ""

  OMNI_VENV="$SHARED_OMNI/tools/omnistat/venv/bin"
  for cmd in omnistat-usermode omnistat-query omnistat-inspect; do
    if [[ -x "$OMNI_VENV/$cmd" ]]; then
      echo "- $cmd: present"
    else
      echo "- $cmd: **MISSING**"
    fi
  done

  echo ""
  echo "### Site config (key knobs)"
  if [[ -f "$SHARED_OMNI/tools/omnistat/omnistat.config" ]]; then
    cfg="$SHARED_OMNI/tools/omnistat/omnistat.config"
  elif [[ -f "$SHARED_OMNI/tools/omnistat/omnihub.config" ]]; then
    cfg="$SHARED_OMNI/tools/omnistat/omnihub.config"
  else
    cfg=""
  fi
  if [[ -n "$cfg" ]]; then
    grep -E 'enable_rocprofiler|enable_amd_smi|enable_kernel_trace|job_detection_file|sampling_mode|counters' \
      "$cfg" || echo "(no matching keys)"
  else
    echo "omnistat.config / omnihub.config not found under $SHARED_OMNI/tools/omnistat/"
  fi

  echo ""
  echo "### Rocprofiler PMC configs"
  ls "$SHARED_OMNI/tools/omnistat-rocprofiler/"*.config 2>/dev/null || ls "$SHARED_OMNI/tools/omnistat-rocprofiler/"omnihub.*.config 2>/dev/null || echo "(none found)"

  echo ""
  echo "### rocprofiler-sdk Python extension"
  if "$OMNI_VENV/python" -c "from omnistat.rocprofiler_sdk_extension import get_samplers" 2>/dev/null; then
    echo "sdk extension: OK"
  else
    echo "sdk extension: failed on login node (may work on compute nodes with ROCm loaded)"
  fi

  echo ""
  echo "### Kernel trace library"
  if [[ -f "$SHARED_OMNI/tools/omnistat/build-trace/libomnistat_trace.so" ]]; then
    echo "libomnistat_trace.so: present"
  else
    echo "libomnistat_trace.so: not found"
  fi

  echo ""
  echo "## ai4science omnistat-pr271 (reference)"
  PR271="$AI4S_SHARED/tools/omnistat-pr271/bin"
  if [[ -d "$AI4S_SHARED/tools/omnistat-pr271" ]]; then
    for cmd in omnistat-usermode omnistat-query omnistat-inspect; do
      if [[ -x "$PR271/$cmd" ]]; then
        echo "- $cmd: present"
      else
        echo "- $cmd: missing"
      fi
    done
  else
    echo "Not installed at $AI4S_SHARED/tools/omnistat-pr271 (HydraGNN launcher creates this)"
  fi

  echo ""
  echo "## Integration recommendation"
  echo "- OmniHub path: use --tools omnistat + omnihub-process -> processed-data/"
  echo "- Deep agent path (omnistat-inspect, PromQL): legacy sbatch_train_perf_amd.sh + omnistat-pr271"
  echo "- If omnistat-inspect missing under /shared/omnihub, agents should not expect inspect JSON from OmniHub jobs"
} | report
