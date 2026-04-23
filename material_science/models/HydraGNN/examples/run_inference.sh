#!/usr/bin/env bash
# run_inference.sh — Load a HydraGNN Predictive GFM 2024 checkpoint and run predictions
#
# Run inside the container launched by docker_run.sh:
#   docker exec -it hydragnn bash /examples/run_inference.sh
#
# Prerequisites:
#   - HydraGNN installed at /workspace/HydraGNN (docker_run.sh handles this)
#   - A checkpoint downloaded from HuggingFace mlupopa/HydraGNN_Predictive_GFM_2024
#     Set HG_CHECKPOINT and HG_CONFIG to matching .pk and config.json files.
#
# OPTIONAL EDITS:
#   HG_CHECKPOINT — path to a .pk checkpoint file (required)
#   HG_CONFIG     — path to the matching config.json (required; same Hub subfolder)
#   HG_OUTPUT_DIR — where to write prediction outputs (default: /workspace/results)

set -euo pipefail

HG_CHECKPOINT="${HG_CHECKPOINT:-}"
HG_CONFIG="${HG_CONFIG:-}"
HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-/workspace/results}"

if [[ -z "${HG_CHECKPOINT}" || -z "${HG_CONFIG}" ]]; then
    echo "ERROR: HG_CHECKPOINT and HG_CONFIG must both be set."
    echo ""
    echo "Download a checkpoint and its matching config.json from HuggingFace:"
    echo "  https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024/tree/main/Ensemble_of_models"
    echo ""
    echo "Example (downloads one trial):"
    echo "  pip install huggingface_hub"
    echo "  python - <<'EOF'"
    echo "  from huggingface_hub import hf_hub_download"
    echo "  import os"
    echo "  repo = 'mlupopa/HydraGNN_Predictive_GFM_2024'"
    echo "  local = '/workspace/checkpoints'"
    echo "  os.makedirs(local, exist_ok=True)"
    echo "  # Replace TRIAL with an actual trial folder name from the Hub tree"
    echo "  hf_hub_download(repo, 'Ensemble_of_models/TRIAL/config.json', local_dir=local)"
    echo "  # Download matching .pk checkpoint file"
    echo "  EOF"
    echo ""
    echo "Then:"
    echo "  HG_CHECKPOINT=/workspace/checkpoints/.../gfm_0.XXX_epoch_YYY.pk \\"
    echo "  HG_CONFIG=/workspace/checkpoints/.../config.json \\"
    echo "  bash /examples/run_inference.sh"
    exit 1
fi

mkdir -p "$HG_OUTPUT_DIR"
cd /workspace/HydraGNN

echo "=== HydraGNN Predictive GFM 2024 — Inference ==="
echo "  Checkpoint : $HG_CHECKPOINT"
echo "  Config     : $HG_CONFIG"
echo "  Output     : $HG_OUTPUT_DIR"
echo ""

python - <<PYEOF
import hydragnn
import torch

config_file = "${HG_CONFIG}"
checkpoint  = "${HG_CHECKPOINT}"
output_dir  = "${HG_OUTPUT_DIR}"

print(f"Loading model from config: {config_file}")
model = hydragnn.load_existing_model(config_file, checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)

# --- Replace the block below with your actual input graph construction ---
# Inputs must match the featurization used during training (see upstream HydraGNN docs).
# Example placeholder: build an ASE Atoms object or a torch_geometric Data object
# and call hydragnn.run_prediction(model, data) per upstream API.
print("")
print("Model loaded successfully. Build your input graph and call:")
print("  hydragnn.run_prediction(model, data)")
print("See recipes/inference/README.md for details.")
PYEOF
