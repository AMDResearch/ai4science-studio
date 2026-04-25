#!/usr/bin/env bash
# run_inference.sh — Optimized SwinUNETR inference with AMP + torch.compile
#
# Run INSIDE the container launched by docker_run.sh (infer image, ROCm 7.0).
# Combined AMP + torch.compile(max-autotune) gives 2.9× speedup vs baseline.
#
# OPTIONAL EDITS:
#   CHECKPOINT  — path to trained .pth checkpoint (required)
#   INPUT_DIR   — directory containing NIfTI volumes to segment (default: /data/test)
#   OUTPUT_DIR  — where to write segmentation masks (default: /workspace/results)
#   ROI_X/Y/Z  — sliding window patch size (default: 96 96 96)
#   USE_COMPILE — set to 1 to enable torch.compile max-autotune (default: 1, ~2.9× total)
#   AMP_DTYPE   — float16 or bfloat16 (default: float16)

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-}"
INPUT_DIR="${INPUT_DIR:-/data/test}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results}"
ROI_X="${ROI_X:-96}"
ROI_Y="${ROI_Y:-96}"
ROI_Z="${ROI_Z:-96}"
USE_COMPILE="${USE_COMPILE:-1}"
AMP_DTYPE="${AMP_DTYPE:-float16}"

mkdir -p "$OUTPUT_DIR"

# --- Check checkpoint ---
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: CHECKPOINT is not set."
    echo "  Set it to your trained checkpoint, e.g.:"
    echo "  CHECKPOINT=/workspace/checkpoints/model_final.pt bash run_inference.sh"
    exit 1
fi

echo "=== SwinUNETR Inference ==="
echo "  Checkpoint   : $CHECKPOINT"
echo "  Input        : $INPUT_DIR"
echo "  Output       : $OUTPUT_DIR"
echo "  ROI size     : ${ROI_X}×${ROI_Y}×${ROI_Z}"
echo "  AMP dtype    : $AMP_DTYPE  (float16 recommended; bfloat16 underperforms)"
echo "  torch.compile: $([ "$USE_COMPILE" == "1" ] && echo "max-autotune (2.9× total speedup)" || echo "disabled")"
echo ""

python - <<PYEOF
import torch
import os
from monai.networks.nets import SwinUNETR
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd, ScaleIntensityRanged, ToTensord
)
from pathlib import Path

checkpoint = "$CHECKPOINT"
input_dir  = "$INPUT_DIR"
output_dir = "$OUTPUT_DIR"
roi_size   = (int("$ROI_X"), int("$ROI_Y"), int("$ROI_Z"))
use_compile = "$USE_COMPILE" == "1"
amp_dtype   = torch.float16 if "$AMP_DTYPE" == "float16" else torch.bfloat16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model = SwinUNETR(
    img_size=roi_size,
    in_channels=1,
    out_channels=2,
    feature_size=int(os.environ.get("FEATURE_SIZE", "48")),
    use_checkpoint=True,
)
state = torch.load(checkpoint, map_location=device)
model.load_state_dict(state.get("state_dict", state))
model.eval().to(device)

# Apply torch.compile for inference speedup
if use_compile:
    print("Compiling model with max-autotune (first inference will be slow) ...")
    model = torch.compile(model, mode="max-autotune")

inferer = SlidingWindowInferer(
    roi_size=roi_size,
    sw_batch_size=4,
    overlap=0.5,
    mode="gaussian",
)

import nibabel as nib
import numpy as np

nifti_files = sorted(Path(input_dir).glob("*.nii.gz"))
if not nifti_files:
    print(f"No .nii.gz files found in {input_dir}")
    raise SystemExit(1)

print(f"Running inference on {len(nifti_files)} volumes ...")
for nii_path in nifti_files:
    nii = nib.load(str(nii_path))
    img = torch.tensor(nii.get_fdata()[None, None], dtype=torch.float32).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = inferer(img, model)

    mask = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    out_path = Path(output_dir) / nii_path.name.replace(".nii.gz", "_seg.nii.gz")
    nib.save(nib.Nifti1Image(mask, nii.affine), str(out_path))
    print(f"  Saved: {out_path}")

print(f"\nSegmentation complete. Results in: {output_dir}")
PYEOF
