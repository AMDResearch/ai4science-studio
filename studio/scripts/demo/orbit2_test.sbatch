#!/bin/bash
#SBATCH --job-name=orbit2-test
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=logs/orbit2_test_%j.log
#SBATCH --error=logs/orbit2_test_%j.log
# Smoke-test the ORBIT-2 synthetic inference path the studio uses. Submit from
# inside the checkout with AI4S_SHARED_DIR set (sbatch exports your environment).

export AI4S_SHARED_DIR="${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR}"
export ORBIT2_SIF="${ORBIT2_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
export ORBIT2_OVERLAY="${ORBIT2_OVERLAY:-${AI4S_SHARED_DIR}/models/ORBIT-2/overlays/orbit2-overlay.img}"
export ORBIT2_ROOT="${ORBIT2_ROOT:-${AI4S_SHARED_DIR}/models/ORBIT-2/code/ORBIT-2}"
export ORBIT2_USE_SYNTHETIC=1
export ORBIT2_CONFIG=interm_8m_synthetic_1gpu.yaml
REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
export STUDIO_EXAMPLES_DIR="$REPO/earth_science/models/ORBIT-2/examples"

bash "$STUDIO_EXAMPLES_DIR/sbatch_infer_amd.sh"
