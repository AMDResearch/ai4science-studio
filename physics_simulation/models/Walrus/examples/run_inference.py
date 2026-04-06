#!/usr/bin/env python3
"""
Walrus inference — autoregressive rollout from a pre-trained checkpoint.

Environment variables (all optional):
  WALRUS_WEIGHTS_DIR   local dir with HF weights (default: ./walrus-weights)
  WALRUS_INPUT         path to input field file (required for custom data)
  WALRUS_STEPS         number of rollout steps (default: 50)
  WALRUS_OUTPUT        output file path (default: outputs/walrus_rollout.pt)

Usage:
  python run_inference.py
  WALRUS_STEPS=100 WALRUS_INPUT=/data/ic.pt python run_inference.py
"""

import os
import sys
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEIGHTS_DIR = os.environ.get("WALRUS_WEIGHTS_DIR", "./walrus-weights")
INPUT       = os.environ.get("WALRUS_INPUT", "")
STEPS       = int(os.environ.get("WALRUS_STEPS", "50"))
OUTPUT      = os.environ.get("WALRUS_OUTPUT", "outputs/walrus_rollout.pt")

print("=== Walrus Inference ===")
print(f"  Weights dir : {WEIGHTS_DIR}")
print(f"  Input       : {INPUT or '(demo — random noise)'}")
print(f"  Steps       : {STEPS}")
print(f"  Output      : {OUTPUT}")
print()

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected — running on CPU (slow)")

# ---------------------------------------------------------------------------
# Load model from HuggingFace
# ---------------------------------------------------------------------------
print("Loading Walrus model …")
try:
    from huggingface_hub import snapshot_download  # type: ignore
except ImportError:
    print("error: huggingface-hub not installed. Run: pip install huggingface-hub", file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(WEIGHTS_DIR):
    print(f"Downloading weights to {WEIGHTS_DIR} …")
    snapshot_download("polymathic-ai/walrus", local_dir=WEIGHTS_DIR)

# Load via upstream walrus package
try:
    import walrus as walrus_pkg  # type: ignore
    model = walrus_pkg.load_model(WEIGHTS_DIR)
except (ImportError, AttributeError):
    # Fallback: load via torch.load if the package API differs
    print("Falling back to torch.load …")
    ckpt = torch.load(os.path.join(WEIGHTS_DIR, "model.pt"), map_location=device)
    model = ckpt

model = model.to(device)
model.eval()
print("Model loaded.")

# ---------------------------------------------------------------------------
# Prepare input
# ---------------------------------------------------------------------------
if INPUT and os.path.exists(INPUT):
    print(f"Loading input from {INPUT} …")
    u = torch.load(INPUT, map_location=device)
else:
    print("No WALRUS_INPUT provided — using random noise as demo input.")
    # Shape: (batch=1, history=4, channels=1, height=64, width=64)
    u = torch.randn(1, 4, 1, 64, 64, device=device)

print(f"  Input shape: {u.shape}")

# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT) if os.path.dirname(OUTPUT) else ".", exist_ok=True)

print(f"Running {STEPS}-step rollout …")
predictions = [u[:, -1:].cpu()]  # store last frame of each step

with torch.no_grad():
    current = u
    for step in range(STEPS):
        u_next = model(current)
        predictions.append(u_next[:, -1:].cpu())
        # Slide the history window
        current = torch.cat([current[:, 1:], u_next[:, -1:]], dim=1)
        if (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{STEPS}")

rollout = torch.cat(predictions, dim=1)
print(f"Saving rollout to {OUTPUT} … (shape: {rollout.shape})")
torch.save(rollout, OUTPUT)
print("Done.")
