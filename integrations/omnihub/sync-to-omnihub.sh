#!/usr/bin/env bash
# Sync ai4science OmniHub application stubs into the OmniHub checkout.
#
# OmniHub requires --app-config paths relative to the omnihub repo root.
# Source of truth lives here; this script copies into $OMNIHUB_DIR/applications/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AI4S_REPO_ROOT="${AI4S_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OMNIHUB_DIR="${OMNIHUB_DIR:-}"

if [[ -z "$OMNIHUB_DIR" ]]; then
  echo "ERROR: set OMNIHUB_DIR to your OmniHub checkout (e.g. export OMNIHUB_DIR=\$HOME/git/omnihub)" >&2
  exit 2
fi

if [[ ! -x "$OMNIHUB_DIR/omnihub-generate-job" ]]; then
  echo "ERROR: OMNIHUB_DIR does not look like an OmniHub repo: $OMNIHUB_DIR" >&2
  exit 2
fi

SRC="$SCRIPT_DIR/applications"
DEST="$OMNIHUB_DIR/applications"

mkdir -p "$DEST"

for app_dir in "$SRC"/*/; do
  [[ -d "$app_dir" ]] || continue
  name=$(basename "$app_dir")
  target="$DEST/ai4science-${name}"
  echo "Syncing $name -> $target"
  rsync -a --delete "$app_dir" "$target/"
done

echo "Done. Use --app-config applications/ai4science-<name>/config.yaml with omnihub-generate-job."
