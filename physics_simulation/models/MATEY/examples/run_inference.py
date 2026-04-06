#!/usr/bin/env python3
"""
MATEY inference — autoregressive rollout from a trained checkpoint.

Environment variables (all optional):
  MATEY_CHECKPOINT   path to .pt checkpoint file (required)
  MATEY_CONFIG       path to YAML config used during training (required)
  MATEY_INPUT        path to HDF5 initial-condition file (required)
  MATEY_STEPS        number of rollout steps (default: 100)
  MATEY_OUTPUT       output HDF5 path (default: outputs/rollout.h5)

Usage:
  python run_inference.py
  MATEY_CHECKPOINT=/checkpoints/model.pt \
  MATEY_INPUT=/data/ic.h5 \
  MATEY_STEPS=200 python run_inference.py
"""

import os
import sys
import h5py
import torch
import yaml
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT = os.environ.get("MATEY_CHECKPOINT", "")
CONFIG     = os.environ.get("MATEY_CONFIG", "")
INPUT      = os.environ.get("MATEY_INPUT", "")
STEPS      = int(os.environ.get("MATEY_STEPS", "100"))
OUTPUT     = os.environ.get("MATEY_OUTPUT", "outputs/rollout.h5")

print("=== MATEY Inference ===")
print(f"  Checkpoint : {CHECKPOINT}")
print(f"  Config     : {CONFIG}")
print(f"  Input      : {INPUT}")
print(f"  Steps      : {STEPS}")
print(f"  Output     : {OUTPUT}")
print()

# Validate required inputs
missing = []
if not CHECKPOINT:
    missing.append("MATEY_CHECKPOINT")
if not CONFIG:
    missing.append("MATEY_CONFIG")
if not INPUT:
    missing.append("MATEY_INPUT")

if missing:
    print("error: the following environment variables must be set:", file=sys.stderr)
    for var in missing:
        print(f"  {var}", file=sys.stderr)
    sys.exit(1)

for path, label in [(CHECKPOINT, "checkpoint"), (CONFIG, "config"), (INPUT, "input")]:
    if not os.path.exists(path):
        print(f"error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected — running on CPU (slow)")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading model …")
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

# MATEY model loading — import from installed matey package
try:
    from matey import build_model  # type: ignore
except ImportError:
    print("error: matey package not found. Install with: pip install -e /matey", file=sys.stderr)
    sys.exit(1)

model = build_model(cfg)
state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state.get("model_state_dict", state))
model.to(device)
model.eval()
print("Model loaded.")

# ---------------------------------------------------------------------------
# Load initial condition
# ---------------------------------------------------------------------------
print("Loading initial condition …")
with h5py.File(INPUT, "r") as f:
    # Assumes the IC is stored under a 'field' key — adjust if needed
    key = list(f.keys())[0]
    ic = torch.tensor(f[key][0:1], dtype=torch.float32).to(device)
print(f"  IC shape: {ic.shape}")

# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT) if os.path.dirname(OUTPUT) else ".", exist_ok=True)

print(f"Running {STEPS}-step rollout …")
predictions = [ic.cpu().numpy()]
current = ic

with torch.no_grad():
    for step in range(STEPS):
        current = model(current)
        predictions.append(current.cpu().numpy())
        if (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{STEPS}")

# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
print(f"Saving rollout to {OUTPUT} …")
rollout = np.concatenate(predictions, axis=0)
with h5py.File(OUTPUT, "w") as f:
    f.create_dataset("rollout", data=rollout, compression="gzip")

print(f"Done. Output shape: {rollout.shape}")
