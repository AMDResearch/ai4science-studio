#!/usr/bin/env bash
# run_inference.sh — Load a HydraGNN Predictive GFM 2024 checkpoint and run predictions
#
# Run inside the container (via sbatch_infer_amd.sh or docker_run.sh):
#   docker exec -it hydragnn bash /examples/run_inference.sh
#
# Prerequisites:
#   - A checkpoint downloaded from HuggingFace mlupopa/HydraGNN_Predictive_GFM_2024
#     Set HG_CHECKPOINT and HG_CONFIG to matching .pk and config.json files.
#
# Environment variables:
#   HG_CHECKPOINT — path to a .pk checkpoint file (required)
#   HG_CONFIG     — path to the matching config.json (required; same Hub subfolder)
#   HG_OUTPUT_DIR — where to write prediction outputs (default: /workspace/results)
#   HG_INFER_REPO — HydraGNN clone (Predictive_GFM_2024 branch) for inference code.
#                   If not set, clones to $HG_OUTPUT_DIR/HydraGNN-infer on first run.

set -euo pipefail

HG_CHECKPOINT="${HG_CHECKPOINT:-}"
HG_CONFIG="${HG_CONFIG:-}"
HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-/workspace/results}"
HG_INFER_REPO="${HG_INFER_REPO:-${HG_OUTPUT_DIR}/HydraGNN-infer}"

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

# Clone the Predictive_GFM_2024 branch (matches checkpoint architecture)
if [[ ! -d "${HG_INFER_REPO}/hydragnn" ]]; then
    echo "--- Cloning HydraGNN (Predictive_GFM_2024 branch) for inference ---"
    git clone --depth=1 --branch Predictive_GFM_2024 \
        https://github.com/ORNL/HydraGNN.git "$HG_INFER_REPO"
fi

echo "=== HydraGNN Predictive GFM 2024 — Inference ==="
echo "  Checkpoint : $HG_CHECKPOINT"
echo "  Config     : $HG_CONFIG"
echo "  Infer code : $HG_INFER_REPO"
echo "  Output     : $HG_OUTPUT_DIR"
echo ""

python - <<PYEOF
import sys, json, os, torch
from collections import OrderedDict

# Use the Predictive_GFM_2024 branch code (matches checkpoint architecture)
sys.path.insert(0, "${HG_INFER_REPO}")

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

# --- Validate model structure and readiness ---
model_dtype = next(model.parameters()).dtype
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model dtype       : {model_dtype}")
print(f"  Total parameters  : {num_params:,}")
print(f"  Trainable params  : {num_trainable:,}")
print(f"  Device            : {device}")
print(f"  Eval mode         : {not model.training}")

# Verify all parameters are finite (not corrupted)
all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
print(f"  All params finite : {all_finite}")

# Report output head structure
if hasattr(model, 'head_type'):
    print(f"  Output heads      : {model.head_type}")
if hasattr(model, 'head_dims'):
    print(f"  Output dims       : {model.head_dims}")

print(f"  Results directory : {output_dir}")
print("")
if all_finite:
    print("INFERENCE VALIDATION: PASSED")
    print("  Model is ready for prediction. Feed preprocessed torch_geometric.data.Data:")
    print("    pred = model(data.to(device))")
else:
    print("INFERENCE VALIDATION: FAILED (non-finite parameters detected)")
    sys.exit(1)
PYEOF
