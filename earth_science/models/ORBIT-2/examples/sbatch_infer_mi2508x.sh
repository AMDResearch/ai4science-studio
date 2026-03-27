#!/usr/bin/env bash
# ORBIT-2 visualization / inference-style run on AMD HPC Fund-style SLURM (partition mi2508x).
#
# Prerequisites:
#   1. Clone https://github.com/XiaoWang-Github/ORBIT-2 and install per upstream README (AMD + ROCm).
#   2. Stage data; set low_res_dir / high_res_dir in the YAML to your paths.
#   3. Download matching .ckpt (+ YAML if needed) from https://huggingface.co/jychoi-hpc/ORBIT-2
#   4. Export ORBIT2_ROOT to the clone root. Optional: ORBIT2_CONFIG (default interm_8m_ft.yaml),
#      ORBIT2_CHECKPOINT to pass --checkpoint, STUDIO_ORBIT2_LAUNCHER to override this run_visualize.py path.
#   5. YAML parallelism must match the number of SLURM tasks (here: 8 tasks, 8 GPUs).
#
# See ../recipes/inference-and-visualization.md and ../recipes/local-cluster-amd.md

#SBATCH -A YOUR_PROJECT_HERE
#SBATCH -J orbit2-vis
#SBATCH --partition=mi2508x
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -o orbit2-vis-%j.out
#SBATCH -e orbit2-vis-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="${STUDIO_ORBIT2_LAUNCHER:-$SCRIPT_DIR/run_visualize.py}"

if [[ -z "${ORBIT2_ROOT:-}" ]]; then
  echo "error: export ORBIT2_ROOT to your ORBIT-2 clone path" >&2
  exit 2
fi

# shellcheck disable=SC1091
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate orbit

# module load rocm  # if your site uses modules; align with your PyTorch ROCm build

export PYTHONNOUSERSITE=1
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/orbit2-miopen-${SLURM_JOB_ID:-$$}"
mkdir -p "$MIOPEN_USER_DB_PATH"

CONFIG="${ORBIT2_CONFIG:-interm_8m_ft.yaml}"
EXTRA=()
if [[ -n "${ORBIT2_CHECKPOINT:-}" ]]; then
  EXTRA+=(--checkpoint "$ORBIT2_CHECKPOINT")
fi

# Optional: Slingshot / RCCL tuning on Cray systems only — uncomment if your admins document it.
# export FI_MR_CACHE_MONITOR=kdreg2

echo "LAUNCHER=$LAUNCHER ORBIT2_ROOT=$ORBIT2_ROOT CONFIG=$CONFIG SLURM_NTASKS=${SLURM_NTASKS:-unset}"

srun "${PYTHON:-python}" "$LAUNCHER" --orbit2-root "$ORBIT2_ROOT" "$CONFIG" "${EXTRA[@]}"
