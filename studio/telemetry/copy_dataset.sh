#!/usr/bin/env bash
# Copy the Alexandria-v2.bp HydraGNN training dataset (~23 GB) from the colleague's
# tree into spannala-owned storage, so live training does not depend on /shared/aaji.
#
# Idempotent: rsync only transfers changed/missing files. Safe to re-run.
set -euo pipefail

SRC="/shared/aaji/models/HydraGNN/weights/Alexandria-v2.bp"
DST_DIR="/shared/spannala/models/HydraGNN/weights"
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
