#!/bin/bash
#SBATCH --job-name=tele-diag2
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:06:00
#SBATCH --output=/shared/spannala/models/HydraGNN/train_work/logs/tele_diag2_%j.log
#SBATCH --error=/shared/spannala/models/HydraGNN/train_work/logs/tele_diag2_%j.log
set -uo pipefail
VENV=/shared/spannala/perf-tools/omnistat-venv
DIR=/shared/spannala/models/HydraGNN/train_work/perf-runs/diag2-${SLURM_JOB_ID}
mkdir -p "$DIR"; cd "$DIR"

# Render two configs: rocprofiler ON (as-is) and OFF.
TMPL=/home/spannala/Projects/ai4science-studio/studio/telemetry/omnistat.config.template
render() { sed -e "s|@JOB_DIR@|$DIR|g" -e "s|@PERF_TOOLS_DIR@|/shared/spannala/perf-tools|g" "$TMPL" > "$1"; }
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
