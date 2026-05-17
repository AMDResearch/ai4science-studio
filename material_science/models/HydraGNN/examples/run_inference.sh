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

echo "=== HydraGNN Predictive GFM 2024 — Inference ==="
echo "  Checkpoint : $HG_CHECKPOINT"
echo "  Config     : $HG_CONFIG"
echo "  Output     : $HG_OUTPUT_DIR"
echo ""

python - <<PYEOF
import json, os, torch
from collections import OrderedDict

config_file = "${HG_CONFIG}"
checkpoint  = "${HG_CHECKPOINT}"
output_dir  = "${HG_OUTPUT_DIR}"

os.makedirs(output_dir, exist_ok=True)

with open(config_file) as f:
    config = json.load(f)

from hydragnn.models.create import create_model_config
from hydragnn.utils.distributed import get_device_name

print(f"Creating model from config: {config_file}")
model = create_model_config(
    config=config["NeuralNetwork"],
    verbosity=config["Verbosity"]["level"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)

print(f"Loading checkpoint: {checkpoint}")
map_location = {"cuda:0": str(device)}
ckpt = torch.load(checkpoint, map_location=map_location)
state_dict = ckpt["model_state_dict"]
# GFM checkpoints were saved with DDP (keys prefixed "module.") but we infer
# without DDP wrapping, so strip the prefix before load_state_dict.
if next(iter(state_dict)).startswith("module."):
    state_dict = OrderedDict((k[len("module."):], v) for k, v in state_dict.items())
model.load_state_dict(state_dict)
model.eval()

print("")
print("Model loaded successfully.")
print("Build your input graph (torch_geometric.data.Data) and call:")
print("  pred = model(data.to(device))")
print(f"Results directory: {output_dir}")
PYEOF
