#!/bin/bash
# AI4Science Studio — 4-day compute-node holder allocation.
# Set the partition/account at submit time, e.g.
#   SBATCH_PARTITION=<partition> SBATCH_ACCOUNT=<account> sbatch studio-hold-4d.sbatch
# (sbatch environment variables override the placeholder #SBATCH lines below).
# Holds an MI355X node for 4 days so the Studio backend/frontend can attach to it via
# `srun --jobid=<this> --overlap`. See studio/SERVING.md for how services attach.
#SBATCH --job-name=studio-hold
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/studio-hold-%j.log

echo "[hold] studio 4-day allocation on $(hostname) starting at $(date)"
echo "[hold] Job $SLURM_JOB_ID holds this node until +4 days. Attach services with:"
echo "[hold]   srun --jobid=$SLURM_JOB_ID --overlap ..."
# Keep the allocation alive for the full 4 days (4*24*3600 = 345600 s).
sleep 345600
echo "[hold] studio 4-day allocation ending at $(date)"
