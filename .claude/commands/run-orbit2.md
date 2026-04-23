# Run ORBIT-2 inference/visualization on an AMD cluster

Guide the user through running ORBIT-2 end-to-end on an AMD cluster via SLURM.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. Container runtime**
Which container runtime do you want to use?
- **Apptainer** (recommended for HPC — supports overlays, `srun --mpi=pmix` for multi-GPU, `--rocm` flag)
- **Docker** (simpler setup, no overlay needed, uses `torchrun` for multi-GPU instead of `srun`, single-node only)

**Q1. Upstream repo**
Do you have the ORBIT-2 upstream code cloned locally? If yes, what is the full path (`ORBIT2_ROOT`)? If no, I will generate the clone command.

**Q2. (Apptainer only) SIF path**
Do you have an Apptainer SIF built from `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`? If yes, what is the full path? If no, I will generate the pull command.

**Q3. (Apptainer only) Overlay**
Do you have a pre-built ORBIT-2 overlay image (`orbit2-overlay.img`)? If yes, what is the full path? If no, would you like to build one now (one-time ~15 min job that skips dep install on every future run), or skip the overlay and pay the ~15 min install cost per job?

**Q4. Data mode**
Which data mode do you want to use?
- **Synthetic** — generates a small ~2 MB dataset automatically, downloads the smallest checkpoint from HuggingFace. No real data needed. Use this for a smoke-test.
- **Real data** — requires ERA5/PRISM data staged on the cluster. You will need to provide the config YAML path and checkpoint path.

**Q5. (Real data only) Config and checkpoint**
If real data: what is the config YAML basename (e.g. `interm_8m_ft.yaml`) and the full path to your `.ckpt` checkpoint file?

**Q6. GPU count**
How many GPUs do you want to use?
- **1 GPU** — quick smoke-test; requires editing the SBATCH header (`--gres=gpu:1 --ntasks-per-node=1`).
- **8 GPU** — full distributed run on one node; uses the default SBATCH header.
- **Multi-node** — (Apptainer only) how many nodes? (sets `--nodes=N --ntasks=N*8`)

**Q7. Partition and account**
What is your SLURM partition name and account/project name?

---

## Step 2 — Act on answers

### Common (both runtimes)

**If upstream repo is missing:**
```bash
git clone https://github.com/XiaoWang-Github/ORBIT-2.git /path/to/ORBIT-2
export ORBIT2_ROOT=/path/to/ORBIT-2
```

### Apptainer path

**If SIF is missing:**
```bash
apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
```
Tell the user to set `ORBIT2_SIF` to the resulting `.sif` path.

**If overlay is missing and user wants to build it:**
```bash
export ORBIT2_SIF=<path>
# Edit #SBATCH --partition and --account in build_overlay_amd.sh first
sbatch earth_science/models/ORBIT-2/examples/build_overlay_amd.sh
```
Tell the user to wait for the build job, note the overlay path from the log, then re-invoke with `ORBIT2_OVERLAY` set.

**Edit the SBATCH header** in `earth_science/models/ORBIT-2/examples/sbatch_infer_amd.sh`:
- Set `--partition` and `-A` to the user's values
- If 1 GPU: set `--gres=gpu:1 --ntasks-per-node=1`
- If multi-node: set `--nodes=N --ntasks=N*8`

### Docker path

**Edit the SBATCH header** in `earth_science/models/ORBIT-2/examples/sbatch_infer_docker.sh` to set the user's partition.

Docker-specific notes:
- No SIF or overlay needed — deps are installed at container start
- Uses a single Docker container with all GPUs + `torchrun --standalone --nproc_per_node=N` for distributed launch (instead of `srun --mpi=pmix` with per-rank Apptainer containers)
- A Python rank wrapper maps torchrun env vars (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) → SLURM env vars (`SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`) for upstream compatibility
- A `mpi4py` stub is injected because `distdataset.py` imports `mpi4py` at module level, but Docker has no OpenMPI runtime. The stub is safe because `ORBIT_USE_DDSTORE=0` (default) means no actual MPI calls are made.
- **Multi-node is NOT supported** with Docker — use Apptainer for multi-node runs
- `LD_LIBRARY_PATH` is appended inside the container script (not via `-e`) to preserve `/opt/rocm/lib`

## Step 3 — Submit

### Apptainer
```bash
export ORBIT2_ROOT=<path>
export ORBIT2_SIF=<path>
export ORBIT2_OVERLAY=<path>          # omit if no overlay
export ORBIT2_USE_SYNTHETIC=1         # if synthetic mode
export ORBIT2_CONFIG=<yaml>           # if real data mode
export ORBIT2_CHECKPOINT=<ckpt>       # if real data mode (or omit to auto-download)
sbatch earth_science/models/ORBIT-2/examples/sbatch_infer_amd.sh
```

### Docker
```bash
export ORBIT2_ROOT=<path>
export ORBIT2_USE_SYNTHETIC=1         # if synthetic mode
export ORBIT2_CONFIG=<yaml>           # if real data mode
export ORBIT2_CHECKPOINT=<ckpt>       # if real data mode (or omit to auto-download)
export ORBIT2_NPROC=8                 # GPUs per node (default: 8)
sbatch earth_science/models/ORBIT-2/examples/sbatch_infer_docker.sh
```

## Step 4 — Monitor

```bash
squeue -j <job_id>
tail -f orbit2-*-<job_id>.out
```

Look for `PSNR` and `SSIM` values in the log to confirm successful completion.

## Expected results (synthetic smoke-test)

| GPUs | Wall time | PSNR | SSIM |
|---|---|---|---|
| 1 | ~1–2 min | 14.35 | 0.039 |
| 8 | ~2 min | 14.35 | 0.039 |

## Arguments

$ARGUMENTS
