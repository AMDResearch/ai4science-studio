#!/usr/bin/env bash
# run_tl.sh — Run REINVENT4 Transfer Learning inside the container
#
# Run inside the container: docker exec -it reinvent4 bash /workspace/run_tl.sh
#
# OPTIONAL EDITS:
#   CONFIG_FILE   — path to TOML config (default: /workspace/tl_config.toml)
#   RESULTS_DIR   — log and output directory (default: /workspace/results)
#
# First-time setup:
#   cp /workspace/tl_config.toml.template /workspace/tl_config.toml
#   # Edit model_file and input_smiles_file paths in tl_config.toml

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-/workspace/tl_config.toml}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"

# --- Check config file ---
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Create one from the template:"
    echo "  cp /workspace/tl_config.toml.template /workspace/tl_config.toml"
    echo "  # Then set model_file and input_smiles_file in tl_config.toml"
    exit 1
fi

mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/tl_run.log"

echo "=== REINVENT4 Transfer Learning ==="
echo "  Config  : $CONFIG_FILE"
echo "  Log     : $LOG_FILE"
echo ""
echo "Tip: Add AMP to reinvent/runmodes/TL/learning.py for 10-60% speedup."
echo "     See AMD ROCm blog for the patch snippet."
echo ""

cd /workspace/REINVENT4

reinvent -l "$LOG_FILE" "$CONFIG_FILE"

echo ""
echo "Transfer learning complete."
echo "  Log           : $LOG_FILE"
echo "  Fine-tuned model: check output_model_file in $CONFIG_FILE"
