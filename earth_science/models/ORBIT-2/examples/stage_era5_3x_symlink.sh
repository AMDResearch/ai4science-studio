#!/usr/bin/env bash
# stage_era5_3x_symlink.sh — triple ORBIT-2 ERA5 train samples for HBM-saturation
# perf studies by symlinking the existing year's shards under new year prefixes.
#
# WHY: the Bayes-CAST IterDataModule globs `train/*.npz` (no year filter; see
# src/climate_learn/data/itermodule.py:133). The staged 1.0deg tree ships only
# year 1979 (20 shards x 438 timesteps), which caps the effective per-epoch
# sample count (~1704) and prevents bf16 from reaching >~34% HBM. Adding more
# shard *files* linearly increases samples/epoch and the achievable batch.
#
# This duplicates 1979 data under 1980_/1981_ via symlinks (near-zero disk).
# *** PERF / THROUGHPUT / HBM STUDY ONLY — the data repeats, so it is NOT valid
# for scientific training/eval. *** For real multi-year training, download
# additional years from the ORBIT-2 dataset (see recipes/.../STAGING_ERA5_FOR_HBM.md).
#
# Usage:
#   bash earth_science/models/ORBIT-2/examples/stage_era5_3x_symlink.sh \
#        [DATA_ROOT] [N_COPIES]
#   DATA_ROOT default: $AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
#   N_COPIES  default: 3  (1979 + 2 symlinked years => 3x)
#
# Idempotent: re-running relinks the same targets. Undo: remove the symlinked
# shards (see UNSTAGE note printed at the end).

set -euo pipefail

DATA_ROOT="${1:-${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR}/models/ORBIT-2/data/superres/era5/1.0_deg}"
N_COPIES="${2:-3}"
TRAIN_DIR="${DATA_ROOT}/train"

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "ERROR: train dir not found: $TRAIN_DIR" >&2
  exit 2
fi

# Discover the base year(s) actually present as real (non-symlink) shards.
mapfile -t BASE_SHARDS < <(find "$TRAIN_DIR" -maxdepth 1 -type f -name '*_*.npz' ! -name 'climatology.npz' -printf '%f\n' | sort)
if [[ ${#BASE_SHARDS[@]} -eq 0 ]]; then
  echo "ERROR: no real base shards (YEAR_IDX.npz) in $TRAIN_DIR" >&2
  exit 2
fi

# Base year = prefix of the first shard (e.g. 1979 from 1979_0.npz).
BASE_YEAR="${BASE_SHARDS[0]%%_*}"
echo "Base year detected: ${BASE_YEAR} (${#BASE_SHARDS[@]} real shards)"

made=0
for ((c = 1; c < N_COPIES; c++)); do
  NEW_YEAR=$((BASE_YEAR + c))
  for f in "${BASE_SHARDS[@]}"; do
    idx="${f#*_}"                       # e.g. 0.npz from 1979_0.npz
    link="${TRAIN_DIR}/${NEW_YEAR}_${idx}"
    target="${TRAIN_DIR}/${f}"
    ln -sfn "$target" "$link"
    made=$((made + 1))
  done
  echo "  linked year ${NEW_YEAR} -> ${BASE_YEAR} (${#BASE_SHARDS[@]} shards)"
done

TOTAL=$(find "$TRAIN_DIR" -maxdepth 1 \( -type f -o -type l \) -name '*_*.npz' ! -name 'climatology.npz' | wc -l)
echo "Done. ${made} symlinks created; train/ now has ${TOTAL} shards (~${N_COPIES}x samples)."
echo "UNSTAGE: find '${TRAIN_DIR}' -maxdepth 1 -type l -name '*_*.npz' -delete"
