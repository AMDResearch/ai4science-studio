# Run StormCast deterministic inference on an AMD cluster

Guide the user through running StormCast end-to-end: SIF setup, optional overlay build, and SLURM job submission. StormCast fetches HRRR/GFS data live from NOAA — no dataset pre-staging required.

## Prerequisites check

Ask the user for any missing values, then verify:

1. **`SC_SIF`** — path to an Apptainer SIF built from `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`. If not set or file doesn't exist:
   ```bash
   apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
   ```
2. **`SC_OVERLAY`** (optional but recommended) — path to `stormcast-overlay.img`. If absent, warn the user that each job will run a ~5 min pip install. Offer to build it now:
   ```bash
   export SC_SIF=<path>
   sbatch earth_science/models/StormCast/examples/build_overlay_amd.sh
   # Wait for job to complete, then set SC_OVERLAY to the output path printed in the log
   ```
3. **`SC_START`** — forecast start time in ISO-8601 format, e.g. `2025-01-01T06`. Default: `2025-01-01T06`.
4. **`SC_STEPS`** — number of 1-hour inference steps. Default: `6`.
5. **`SC_OUTPUT`** — output zarr path. Default: auto-derived from `SC_START`. The script removes any existing zarr at that path before running (ZarrBackend does not support overwriting).

## Submission

Once prerequisites are satisfied, construct and run the sbatch command:

```bash
export SC_SIF=<path>
export SC_OVERLAY=<path>          # omit if no overlay
export SC_START=<iso-datetime>
export SC_STEPS=<n>
sbatch earth_science/models/StormCast/examples/sbatch_inference_amd.sh
```

Also remind the user to set `--partition` and `--account` in the SBATCH header if they haven't already — the script has `YOUR_PARTITION_HERE` / `YOUR_ACCOUNT_HERE` placeholders.

## Monitoring and output

After submission:
- `squeue -j <job_id>` to check status
- `tail -f stormcast-infer-<job_id>.out` to follow the log
- On success: output zarr is at `SC_OUTPUT` (default: `outputs/pred-<start>.zarr` next to the script)

## Expected results

| Metric | Value |
|---|---|
| Wall time (with overlay) | ~2–3 min |
| VRAM | ~9.6 GB |
| Output | zarr with u, v, z, t, q, refc, mslp fields |

## Arguments

$ARGUMENTS
