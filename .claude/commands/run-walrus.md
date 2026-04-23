# Run Walrus physics rollout on an AMD cluster

Guide the user through running Walrus autoregressive rollout on AMD GPUs.

## Step 1 — Questionnaire

**Q0. Input data**
Do you have an input field file (`.pt` or `.npy`)? If not, the script will generate random noise as a demo.

**Q1. Rollout steps**
How many steps? Default: 50.

**Q2. Output path**
Where to write the output `.pt` file? Default: `outputs/walrus_rollout.pt`.

**Q3. Weights**
Weights are auto-downloaded from HF (`polymathic-ai/walrus`) on first run. Do you have them locally already? If yes, path?

---

## Step 2 — Launch

### Docker
```bash
cd physics_simulation/models/Walrus/examples
./docker_run.sh inference
```

### Manual
```bash
export WALRUS_STEPS=50
export WALRUS_INPUT=/path/to/input.pt    # optional
python physics_simulation/models/Walrus/examples/run_inference.py
```

## Expected results

Walrus is a standard PyTorch Transformer — runs on ROCm without modification. Weights download (~5 GB) on first run.

Output: `.pt` file with predicted physical field at each timestep.

## Arguments

$ARGUMENTS
