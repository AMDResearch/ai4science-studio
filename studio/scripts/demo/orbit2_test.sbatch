#!/bin/bash
#SBATCH --job-name=orbit2-test
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/home/spannala/orbit2_test_%j.log
#SBATCH --error=/home/spannala/orbit2_test_%j.log

export AI4S_SHARED_DIR=/shared/spannala
export ORBIT2_SIF=/shared/spannala/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
export ORBIT2_OVERLAY=/shared/spannala/models/ORBIT-2/overlays/orbit2-overlay.img
export ORBIT2_ROOT=/shared/aaji/models/ORBIT-2/code/ORBIT-2
export ORBIT2_USE_SYNTHETIC=1
export ORBIT2_CONFIG=interm_8m_synthetic_1gpu.yaml
export STUDIO_EXAMPLES_DIR=/home/spannala/Projects/ai4science-studio/earth_science/models/ORBIT-2/examples

bash /home/spannala/Projects/ai4science-studio/earth_science/models/ORBIT-2/examples/sbatch_infer_amd.sh
