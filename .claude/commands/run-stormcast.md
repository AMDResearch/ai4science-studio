# Run StormCast deterministic inference on an AMD cluster

Guide the user through running StormCast end-to-end on an AMD cluster via SLURM.

## Step 0 — Cluster config check

Before starting, check if `.cluster-config.yaml` (repo root) or `~/.config/ai4science-studio/cluster.yaml` exists. If neither exists, run the `/init-cluster` flow first to auto-discover the cluster environment.

If a config exists, read it and pre-fill Q0 (runtime), Q6 (partition/account) from the saved values. Still present them to the user for confirmation but show the saved defaults.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. Container runtime**
Which container runtime do you want to use?
- **Apptainer** (recommended for HPC — supports overlays, `--rocm` flag for GPU)
- **Docker** (simpler setup, no overlay needed, but env vars must be appended not replaced)

**Q1. (Apptainer only) SIF path**
Do you have an Apptainer SIF built from `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`?
- **Yes** — provide the full path
- **No** — I will generate the pull command
- **Auto-discover** — I will search the filesystem for existing ROCm PyTorch `.sif` files

**Q2. (Apptainer only) Overlay**
Do you have a pre-built StormCast overlay image (`stormcast-overlay.img`)?
- **Yes** — provide the full path
- **No, build one** — one-time ~10 min job that skips a ~5 min pip install on every future run
- **No, skip overlay** — pay the install cost per job
- **Auto-discover** — I will search the filesystem for an existing StormCast overlay image

**Q3. Forecast start time**
What start time do you want to use for the forecast? (ISO-8601 format, e.g. `2025-01-01T06`)

**Q4. Number of steps**
How many 1-hour forecast steps do you want to run?

**Q5. Output path**
Where do you want the output zarr written? (full path, e.g. `/path/to/output/pred.zarr`) Note: any existing zarr at this path will be removed before the run.

**Q6. Partition and account**
How should I determine your SLURM partition and account/project?
- **Provide manually** — type your partition and account names
- **Auto-discover** — I will query SLURM to find available partitions and accounts on this cluster

---

## Step 2 — Act on answers

### Auto-discovery procedures

Run these when the user chose **Auto-discover** for any question. Present the results and let the user confirm or override.

**SIF files (Q1):**
```bash
find "$HOME" /scratch /projects /opt -maxdepth 4 -name "*.sif" 2>/dev/null | head -20
```
Use `$HOME` (not `/home`) so the search works when the home directory is under a non-standard prefix.
Filter results for SIF names containing `rocm` or `pytorch`. Verify with `apptainer inspect <sif>` if multiple candidates.

**Overlay images (Q2):**
```bash
find "$HOME" /scratch /projects /opt -maxdepth 4 -name "*stormcast*overlay*" -o -name "*overlay*stormcast*" 2>/dev/null | head -20
```

**SLURM partition and account (Q6):**
```bash
sinfo -h -o "%P %G" | grep -i gpu
sacctmgr show associations where user=$USER format=account%30,partition%30 -n
```
Present the available GPU partitions and the user's associated accounts. If multiple exist, ask the user to pick.

After auto-discovery, always confirm the found values with the user before proceeding.

---

### Apptainer path

**If SIF is missing:**
```bash
apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
```
Tell the user to set `SC_SIF` to the resulting `.sif` path and re-invoke.

**If overlay is missing and user wants to build it:**
```bash
export SC_SIF=<path>
# Edit #SBATCH --partition and --account in build_overlay_amd.sh first
sbatch earth_science/models/StormCast/examples/build_overlay_amd.sh
```
Tell the user to wait for the build job to complete, note the overlay path from the log, then re-invoke with `SC_OVERLAY` set.

**Edit the SBATCH header** in `earth_science/models/StormCast/examples/sbatch_inference_amd.sh` to set the user's partition and account (replacing `YOUR_PARTITION_HERE` / `YOUR_ACCOUNT_HERE`).

### Docker path

**Edit the SBATCH header** in `earth_science/models/StormCast/examples/sbatch_inference_docker.sh` to set the user's partition and account.

Docker-specific notes:
- No SIF or overlay needed — deps are installed at container start (~5 min)
- Uses `docker run --rm` with GPU device passthrough
- `LD_LIBRARY_PATH` is appended inside the container script (not via `-e`) to preserve `/opt/rocm/lib`
- Wall time is set to 2 hours to allow for first-run model download + inference

## Step 3 — Submit

### Apptainer
```bash
export SC_SIF=<path>
export SC_OVERLAY=<path>        # omit if user chose to skip overlay
export SC_START=<iso-datetime>
export SC_STEPS=<n>
export SC_OUTPUT=<path>
sbatch earth_science/models/StormCast/examples/sbatch_inference_amd.sh
```

### Docker
```bash
export SC_START=<iso-datetime>
export SC_STEPS=<n>
export SC_OUTPUT=<path>
sbatch earth_science/models/StormCast/examples/sbatch_inference_docker.sh
```

## Step 4 — Monitor

```bash
squeue -j <job_id>
tail -f stormcast-*-<job_id>.out
```

On success: output zarr is at the path the user specified. Validate with:
```python
import xarray as xr
ds = xr.open_zarr("<output_path>", consolidated=False)
print(ds)
```

## Expected results

| Runtime | Wall time (with overlay) | Wall time (no overlay) | VRAM | Output |
|---|---|---|---|---|
| Apptainer (GPU) | ~2–3 min | ~7–8 min | ~9.6 GB | zarr with u, v, z, t, q, refc, mslp fields |
| Docker (GPU) | N/A | ~7–8 min | ~9.6 GB | same |

## Arguments

$ARGUMENTS
