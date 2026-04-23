# Run StormCast deterministic inference on an AMD cluster

Guide the user through running StormCast end-to-end on an AMD cluster via SLURM.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. Container runtime**
Which container runtime do you want to use?
- **Apptainer** (recommended for HPC — supports overlays, `--rocm` flag for GPU)
- **Docker** (simpler setup, no overlay needed, but env vars must be appended not replaced)

**Q1. (Apptainer only) SIF path**
Do you have an Apptainer SIF built from `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`? If yes, what is the full path? If no, I will generate the pull command.

**Q2. (Apptainer only) Overlay**
Do you have a pre-built StormCast overlay image (`stormcast-overlay.img`)? If yes, what is the full path? If no, would you like to build one now (one-time ~10 min job that skips a ~5 min pip install on every future run), or skip the overlay and pay the install cost per job?

**Q3. Forecast start time**
What start time do you want to use for the forecast? (ISO-8601 format, e.g. `2025-01-01T06`)

**Q4. Number of steps**
How many 1-hour forecast steps do you want to run?

**Q5. Output path**
Where do you want the output zarr written? (full path, e.g. `/path/to/output/pred.zarr`) Note: any existing zarr at this path will be removed before the run.

**Q6. Partition and account**
What is your SLURM partition name and account/project name?

---

## Step 2 — Act on answers

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
