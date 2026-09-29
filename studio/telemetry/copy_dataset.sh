#!/usr/bin/env bash
# Copy the Alexandria-v2.bp HydraGNN training dataset (~23 GB) from an existing
# staged copy into your own storage, so live training does not depend on someone
# else's tree.
#
# Env:
#   HG_DATASET_SRC  existing Alexandria-v2.bp directory to copy from (required)
#   HG_DATA_DIR     destination directory (default: $AI4S_SHARED_DIR/models/HydraGNN/weights)
#
# Idempotent: rsync only transfers changed/missing files. Safe to re-run.
set -euo pipefail

SRC="${HG_DATASET_SRC:?set HG_DATASET_SRC to an existing Alexandria-v2.bp directory}"
DST_DIR="${HG_DATA_DIR:-${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR or HG_DATA_DIR}/models/HydraGNN/weights}"
DST="${DST_DIR}/Alexandria-v2.bp"

[[ -d "$SRC" ]] || { echo "ERROR: source dataset missing: $SRC" >&2; exit 2; }
mkdir -p "$DST_DIR"

echo "[copy_dataset] rsync $SRC -> $DST"
rsync -a --info=progress2 "$SRC/" "$DST/"

echo "[copy_dataset] verify: source vs dest byte totals"
s=$(du -sb "$SRC" | awk '{print $1}')
d=$(du -sb "$DST" | awk '{print $1}')
echo "  source=$s  dest=$d"
if [[ "$s" == "$d" ]]; then
  echo "[copy_dataset] OK — byte totals match"
else
  echo "[copy_dataset] WARN — byte totals differ (source=$s dest=$d)" >&2
fi

echo "[copy_dataset] dataset groups present:"
ls -1 "$DST"
