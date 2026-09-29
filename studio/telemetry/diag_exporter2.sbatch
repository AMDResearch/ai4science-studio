#!/bin/bash
#SBATCH --job-name=tele-diag2
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:06:00
#SBATCH --output=logs/tele_diag2_%j.log
#SBATCH --error=logs/tele_diag2_%j.log
set -uo pipefail
PERF_TOOLS_DIR="${PERF_TOOLS_DIR:-${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR or PERF_TOOLS_DIR}/perf-tools}"
HG_BASE="${HG_BASE:-${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR or HG_BASE}/models/HydraGNN}"
VENV="${PERF_TOOLS_DIR}/omnistat-venv"
DIR="${HG_BASE}/train_work/perf-runs/diag2-${SLURM_JOB_ID}"

# Omnistat template beside this script (sbatch runs a spool copy, so recover
# the submitted path via scontrol).
_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
TMPL="${OMNISTAT_TEMPLATE:-$(cd "$(dirname "$_CMD")" && pwd)/omnistat.config.template}"
mkdir -p "$DIR"; cd "$DIR"

# Render two configs: rocprofiler ON (as-is) and OFF.
render() { sed -e "s|@JOB_DIR@|$DIR|g" -e "s|@PERF_TOOLS_DIR@|${PERF_TOOLS_DIR}|g" "$TMPL" > "$1"; }
render cfg_on.config
render cfg_off.config
sed -i 's/^enable_rocprofiler = True/enable_rocprofiler = False/' cfg_off.config
sed -i 's/^enable_vendor_counters = True/enable_vendor_counters = False/' cfg_off.config

for tag in on off; do
  echo "============ EXPORTER TEST rocprofiler=$tag ============"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  timeout 15 $VENV/bin/python -m omnistat.standalone --configfile=cfg_${tag}.config \
    --interval 1 --pushinterval 1 --endpoint 127.0.0.1 --port 8101 --log exporter_${tag}.log &
  EPID=$!
  sleep 8
  echo "--- exporter_${tag}.log ---"; cat exporter_${tag}.log 2>/dev/null | head -30
  echo "--- port 8101 responding? ---"
  curl -s http://127.0.0.1:8101/metrics 2>&1 | grep -iE "rocm_|gpu|power|temperature" | head -8 || echo "no metrics endpoint"
  kill $EPID 2>/dev/null; sleep 2
done
echo "=== END DIAG2 ==="
