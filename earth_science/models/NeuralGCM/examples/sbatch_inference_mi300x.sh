#!/usr/bin/env bash
#SBATCH --job-name=neuralgcm-inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/neuralgcm-inference-%j.log
#SBATCH --error=logs/neuralgcm-inference-%j.log
# Adjust --partition and --account for your cluster:
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>

# NeuralGCM deterministic inference — AMD MI300X / SLURM driver
#
# Environment variables (set before sbatch or export in your shell):
#
#   NGC_CHECKPOINT   GCS path suffix (default: v1/deterministic_1_4_deg.pkl)
#   NGC_DATE         Initial condition date YYYY-MM-DD (default: 2020-01-01)
#   NGC_HOUR         Initial condition hour UTC (default: 0)
#   NGC_STEPS        Number of 6h forecast steps (default: 16)
#   NGC_OUTPUT       Output NetCDF path (default: auto)
#   NGC_SEED         JAX PRNG seed for stochastic checkpoints (default: 0)
#   NGC_SIF          Apptainer/Singularity SIF path; if set, runs inside container
#
# Example — bare metal (modules loaded by your site):
#   NGC_DATE=2020-06-15 NGC_STEPS=40 sbatch sbatch_inference_mi300x.sh
#
# Example — Apptainer container:
#   NGC_SIF=/path/to/neuralgcm.sif NGC_STEPS=16 sbatch sbatch_inference_mi300x.sh

set -euo pipefail

NGC_CHECKPOINT="${NGC_CHECKPOINT:-v1/deterministic_1_4_deg.pkl}"
NGC_DATE="${NGC_DATE:-2020-01-01}"
NGC_HOUR="${NGC_HOUR:-0}"
NGC_STEPS="${NGC_STEPS:-16}"
NGC_OUTPUT="${NGC_OUTPUT:-}"
NGC_SEED="${NGC_SEED:-0}"
NGC_SIF="${NGC_SIF:-}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "${SCRIPT_DIR}/logs"

build_args() {
    local args=(
        --checkpoint "${NGC_CHECKPOINT}"
        --date       "${NGC_DATE}"
        --hour       "${NGC_HOUR}"
        --steps      "${NGC_STEPS}"
        --seed       "${NGC_SEED}"
    )
    [[ -n "${NGC_OUTPUT}" ]] && args+=(--output "${NGC_OUTPUT}")
    echo "${args[@]}"
}

echo "=== NeuralGCM inference ==="
echo "  Checkpoint : ${NGC_CHECKPOINT}"
echo "  Date       : ${NGC_DATE}T${NGC_HOUR:02}:00 UTC"
echo "  Steps      : ${NGC_STEPS} × 6h = $((NGC_STEPS * 6))h"
echo "  Seed       : ${NGC_SEED}"

if [[ -n "${NGC_SIF}" ]]; then
    echo "  Container  : ${NGC_SIF}"
    apptainer exec --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${NGC_SIF}" \
        python3 /workspace/run_inference.py $(build_args)
else
    python3 "${SCRIPT_DIR}/run_inference.py" $(build_args)
fi
