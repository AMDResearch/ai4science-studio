#!/usr/bin/env bash
# ORBIT-2 visualization / inference-style run on AMD Instinct via SLURM.
# Thin wrapper around run_infer_amd.sh — all logic lives in that runner.
#
# Quick-start with synthetic data:
#   export ORBIT2_ROOT=/path/to/ORBIT-2-clone
#   export ORBIT2_SIF=${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
#   export ORBIT2_USE_SYNTHETIC=1
#   sbatch sbatch_infer_amd.sh

#SBATCH -J orbit2-vis
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -o orbit2-vis-%j.out
#SBATCH -e orbit2-vis-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

exec bash "$SCRIPT_DIR/run_infer_amd.sh" "$@"
