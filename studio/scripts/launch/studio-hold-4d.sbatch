#!/bin/bash
# AI4Science Studio — 4-day compute-node holder allocation.
# Mirrors the long-running demo allocation (partition lux, account vultr_lux, 1 node).
# Holds an MI355X node for 4 days so the Studio backend/frontend can attach to it via
# `srun --jobid=<this> --overlap`. See studio/SERVING.md for how services attach.
#SBATCH --job-name=studio-hold
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/spannala/Projects/ai4science-studio/studio/scripts/launch/logs/studio-hold-%j.log

echo "[hold] studio 4-day allocation on $(hostname) starting at $(date)"
echo "[hold] Job $SLURM_JOB_ID holds this node until +4 days. Attach services with:"
echo "[hold]   srun --jobid=$SLURM_JOB_ID --overlap ..."
# Keep the allocation alive for the full 4 days (4*24*3600 = 345600 s).
sleep 345600
echo "[hold] studio 4-day allocation ending at $(date)"
