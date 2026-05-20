#!/usr/bin/env bash
# run_train.sh — HydraGNN multi-dataset GFM training (container entrypoint)
#
# This is the single-process / Docker entrypoint for HydraGNN training.
# It clones HydraGNN, symlinks ADIOS datasets, and launches training.
#
# For SLURM multi-node/multi-GPU on AMD: use sbatch_train_amd.sh instead
# (it has its own embedded rank script with MPI support and monkey-patches).
#
# For single-GPU interactive testing inside docker:
#   docker exec -it hydragnn bash /examples/run_train.sh
#
# Environment variables (all optional, sensible defaults provided):
#   HG_REPO_DIR     Path to HydraGNN source clone (default: /workspace/HydraGNN)
#   HG_DATASETS     Comma-separated dataset names (default: ANI1x,Alexandria)
#   HG_DATA_DIR     Directory containing <dataset>-v2.bp dirs (default: /data)
#   HG_BATCH_SIZE   Override batch size from config
#   HG_NUM_EPOCH    Override epoch count from config
#   HG_PRECISION    fp32 | fp64 | bf16 (default: fp64)
#   HG_CONFIG       JSON config filename (default: gfm_mlip.json)

set -euo pipefail

HG_REPO_DIR="${HG_REPO_DIR:-/workspace/HydraGNN}"
HG_DATASETS="${HG_DATASETS:-ANI1x,Alexandria}"
HG_DATA_DIR="${HG_DATA_DIR:-/data}"
HG_BATCH_SIZE="${HG_BATCH_SIZE:-}"
HG_NUM_EPOCH="${HG_NUM_EPOCH:-}"
HG_PRECISION="${HG_PRECISION:-fp64}"
HG_CONFIG="${HG_CONFIG:-gfm_mlip.json}"

EXAMPLE_DIR="${HG_REPO_DIR}/examples/multidataset_hpo_sc26"

# ---------------------------------------------------------------------------
# Clone HydraGNN if not present
# ---------------------------------------------------------------------------
if [[ ! -d "$EXAMPLE_DIR" ]]; then
  echo "--- Cloning HydraGNN ---"
  git clone --depth=1 https://github.com/ORNL/HydraGNN.git "$HG_REPO_DIR"
fi

# ---------------------------------------------------------------------------
# Symlink datasets into expected location
# ---------------------------------------------------------------------------
DATASET_DIR="${EXAMPLE_DIR}/dataset"
mkdir -p "$DATASET_DIR"

IFS=',' read -ra DATASET_ARRAY <<< "$HG_DATASETS"
for ds in "${DATASET_ARRAY[@]}"; do
  target="${DATASET_DIR}/${ds}-v2.bp"
  source="${HG_DATA_DIR}/${ds}-v2.bp"
  if [[ ! -e "$target" ]] && [[ -d "$source" ]]; then
    ln -sfn "$source" "$target"
  fi
done

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------
echo "=== HydraGNN Training ==="
echo "  Repo dir   : $HG_REPO_DIR"
echo "  Example    : $EXAMPLE_DIR"
echo "  Config     : $HG_CONFIG"
echo "  Datasets   : $HG_DATASETS"
echo "  Precision  : $HG_PRECISION"
echo "  Batch size : ${HG_BATCH_SIZE:-config default}"
echo "  Num epochs : ${HG_NUM_EPOCH:-config default}"
echo ""

cd "$EXAMPLE_DIR"

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
TRAIN_ARGS=(
    python -u gfm_mlip_all_mpnn.py
    --inputfile="$HG_CONFIG"
    --multi
    --multi_model_list="$HG_DATASETS"
    --precision="$HG_PRECISION"
    --log="hydragnn-train-$(date +%Y%m%d-%H%M%S)"
)

[[ -n "$HG_BATCH_SIZE" ]] && TRAIN_ARGS+=(--batch_size="$HG_BATCH_SIZE")
[[ -n "$HG_NUM_EPOCH" ]] && TRAIN_ARGS+=(--num_epoch="$HG_NUM_EPOCH")

echo "Running: ${TRAIN_ARGS[*]}"
echo ""
exec "${TRAIN_ARGS[@]}"
