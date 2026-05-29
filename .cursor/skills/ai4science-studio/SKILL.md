---
name: ai4science-studio
description: Applies when working in the AI4Science Studio repository. Describes domain layout, model slug rules, where recipes live, safety expectations, and AMD/ROCm HPC patterns (Apptainer and Docker) validated in practice.
---

# AI4Science Studio (repository)

## Repository map

- **Domains:** `earth_science/` (includes climate and weather), `material_science/`, `protein_folding/`, `healthcare/`, `physics_simulation/`.
- **Models:** `<domain>/models/<model-slug>/` with a `README.md` per model and `recipes/` for that model only.
- **Model index:** Root [`models.yaml`](../../../models.yaml) — machine-readable list of all models. Read this first for discovery.
- **Per-model manifest:** `<domain>/models/<model-slug>/model.yaml` — structured metadata (HF id, license, recipes, env vars, hardware).
- **Human index:** Root [`README.md`](../../../README.md).

## Model slug rule

Hugging Face id `org/model` → directory name `org__model` (replace `/` with double underscore). Document the canonical HF id in the model's `README.md`. Public on-disk names (e.g. `ORBIT-2`, `HydraGNN`) are acceptable when the domain's `models/README.md` allows it.

## Conventions

- Prefer **linking** Hugging Face model cards and upstream GitHub repos instead of vendoring large codebases.
- Do **not** add secrets, API keys, `.env` files with credentials, or PHI to the repo.
- Large artifacts (checkpoints, datasets) belong in `.gitignore` patterns; recipes should explain how to obtain or generate them.

## When adding content

1. Pick the correct **domain** folder.
2. Create or update `models/<model-slug>/README.md` (license, HF id, upstream). If weights are not on Hugging Face (GCS, Google Drive, GitHub releases), set the HF id field to `N/A` and add an "Obtaining model weights" section with a fetch snippet.
3. Place runbooks under `models/<model-slug>/recipes/`. One subfolder per task (`recipes/inference/`, `recipes/finetune/`, etc.), each with its own `README.md` and a callout box at the top linking to `examples/`.
4. Place ready-to-run scripts under `models/<model-slug>/examples/`:
   - `docker_run.sh` — auto-detects AMD Container Toolkit (`docker info | grep -qi amd`) vs device passthrough (`/dev/kfd` + all `/dev/dri/renderD*`); checks for existing container and exits with attach hint; auto-clones upstream repo if absent.
   - `run_<task>.sh` / `run_<task>.py` — all key params overridable via env vars with sensible defaults; prints config summary before running; exits with clear error when required inputs are missing.
   - `preflight_<slug>.py` — smoke-test verifying GPU access and imports.
   - `sbatch_<task>_amd.sh` — SLURM + Apptainer batch script; use `--rocm` (not `--nv`) for AMD GPU passthrough.
   - `sbatch_<task>_docker.sh` — SLURM + Docker batch script; uses device passthrough or `--runtime=amd`.
   - `build_overlay_amd.sh` — (Apptainer only, HPC models with heavy pip deps) builds a persistent ext3 overlay. Run once per cluster; reuse with `--overlay <path>:ro`.
   - All scripts must be `chmod +x`.
5. Create a `model.yaml` in the model folder with structured metadata (name, hf_id, license, task, recipes, env_vars). See existing `model.yaml` files for the schema.
6. Add the model to the root `models.yaml` index.
7. Copy structure from [`_template/`](../../../_template/) when starting a new model folder.
6. For **HPC-oriented** models, consider `recipes/local-cluster-amd.md` and **`data-access.md`** sections on data staging.
7. Write recipes around **AMD Instinct** with **PyTorch ROCm** when that matches upstream supported paths.
8. **GPU arch naming:** Scripts are named `_amd.sh` / `_docker.sh`, not `_mi300x.sh`. The same `rocm7.2.x` image covers MI250X (gfx90a), MI300X (gfx942), and MI350X (gfx950). Only the `ROCM_WHL_TAG` env var and the image tag need to change for a different ROCm generation.

---

# Common HPC patterns (both Apptainer and Docker)

## Rule: always check the Apptainer path first

Before writing or debugging a Docker sbatch script, **always review**:
1. The corresponding `sbatch_*_amd.sh` (Apptainer) script in the repo
2. Any hand-crafted test scripts the user has validated
3. The cluster lessons in memory / SKILL files

Many issues (dep lists, version pins, env vars, torch protection) are already solved in the Apptainer path. Adapt solutions from there rather than reinventing.

## Canonical AMD/ROCm container image

**Use `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`** for all new models unless upstream requires a specific version.

- Covers MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950)
- Python 3.12, PyTorch 2.10.0+rocm7.2.2
- For older hardware (MI100/gfx908): use a `rocm6.x` image
- For future hardware: update image tag + `ROCM_WHL_TAG` together

## ROCm userspace / host driver compatibility

`torch.cuda.is_available()` can return `False` even when `/dev/kfd` and `/dev/dri/renderD*` are visible inside the container. This means the ROCm userspace in the SIF cannot negotiate with the host `amdgpu` kernel module.

**Symptom:** Generation or training runs ~10-20x slower than expected (silent CPU fallback), with no error message.

**Diagnosis:** Check the host driver version (`cat /sys/module/amdgpu/version`) and test inside the container:
```bash
apptainer exec --rocm "$SIF" python3 -c "import torch; print(torch.cuda.is_available())"
```

**Fix (pick one):**
1. Use a newer ROCm SIF whose userspace matches the host driver (e.g. ROCm 7.2.2 for driver 6.16.6)
2. Set `HSA_OVERRIDE_GFX_VERSION=9.4.2` (for gfx942 / MI300A/MI300X) as a workaround

**Known pairings:**
| Host driver (`amdgpu` module) | ROCm 7.0 SIF | ROCm 7.2.2 SIF |
|------|------|------|
| Older (compatible with 7.0) | Works | Works |
| `6.16.6` (MI300A cluster) | **Fails** | Works |

**Best practice:** Add a GPU detection check early in sbatch scripts so users get a clear warning instead of silent CPU fallback:
```bash
GPU_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$GPU_OK" != "True" ]]; then
    echo "WARNING: torch.cuda.is_available() = False — falling back to CPU." >&2
    echo "  Try a newer ROCm SIF or set HSA_OVERRIDE_GFX_VERSION=9.4.2" >&2
fi
```

## Never clobber inherited env vars

Always **append** to `LD_LIBRARY_PATH`, `PYTHONPATH`, `PATH`, `LD_PRELOAD` using `${VAR:-}`. This applies to both runtimes:
- Apptainer `--env LD_LIBRARY_PATH=X` replaces the image default
- Docker `-e LD_LIBRARY_PATH=X` replaces the image default

The `rocm/pytorch` image sets `LD_LIBRARY_PATH=/opt/rocm/lib`. Clobbering it makes `libhsa-runtime64.so` and `libamdhip64.so` unfindable → `torch.cuda.device_count()=0` → silent CPU fallback.

**Correct pattern** — append inside the container script:
```bash
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/my/paths:${PYTHONPATH:-}"
```

Only use `-e` / `--env` for variables you are **introducing** (e.g. `-e ORBIT2_ROOT=/orbit2`).

## SLURM spool dir workaround

`BASH_SOURCE[0]` resolves to `/var/spool/slurmd/jobXXXX/` when SLURM copies the script before execution. Use `scontrol` to recover the original path:

```bash
SCRIPT_DIR=$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')")" && pwd)
```

Applies to both Docker and Apptainer sbatch scripts.

## Python version compatibility

The canonical image is **py3.12**. Some upstream models pin older deps that lack cp312 binary wheels:
- `transformers==4.32.1` → pulls `tokenizers 0.13.x` (no cp312 wheel, requires Rust to build from source). Fix: use `transformers>=4.36,<4.41` (gets tokenizers>=0.15 with cp312 wheels; stays below 4.41 where `transformers.onnx` was removed).
- Always check what Python version the working Apptainer path uses. If it used py3.10, version pins may need adjustment for py3.12.

## ROCm torch protection

After any bulk `pip install`, verify the ROCm torch is still intact:
```bash
python3 -c "import torch; assert 'rocm' in torch.__version__, f'Non-ROCm torch: {torch.__version__}'"
```
Many upstream packages declare `torch` as a dependency. pip may silently pull a CUDA build that shadows the ROCm venv torch.

## Shell quoting rules for sbatch scripts

- Never put unescaped `(`, `)`, `'`, or Python f-strings inside a `bash -c '...'` block — SLURM's shell trips on nested quotes
- For non-trivial Python: write the script to a host file, bind-mount it, invoke with `python3 /scripts/foo.py`
- For heredocs containing Python: use `<< 'PYEOF'` (quoted delimiter) so the shell does not expand `$variables`; pass host variables as `sys.argv` arguments

## HF repo file layout — verify before hardcoding

Use `list_repo_files()` to discover actual filenames. Example: the Walrus HF repo ships `walrus.pt`, not `model.pt`. ORBIT-2 checkpoint names are versioned subdirectories.

```python
from huggingface_hub import list_repo_files
files = [f for f in list_repo_files(repo) if f.endswith(".ckpt")]
chosen = next((f for f in sorted(files) if "8m" in f), sorted(files)[0])
```

## pip-packages / overlay wipe policy

Delete `pip-packages/` or rebuild the overlay entirely when switching to a different base image. Packages compiled for a different Python version or NumPy ABI will crash silently. Stale `.dist-info` also confuses pip resolution.

---

# Apptainer-specific patterns

## Standard Apptainer exec template

```bash
apptainer exec \
    --rocm \
    --overlay "${OVERLAY}:ro" \               # omit if no overlay
    --bind "$WORKSPACE":/workspace \
    "$SIF" bash -c '
source /opt/venv/bin/activate
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
...
'
```

**Rules:**
- Always `--rocm`, never manually bind `/opt/rocm` or `/dev/kfd` — `--rocm` handles GPU device injection and avoids ROCm version mixing
- Never bind-mount host `/opt/rocm` when container ROCm > host ROCm (ABI mismatch crashes RCCL)

Build the SIF once and reuse:
```bash
apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
```

## Overlay build pattern (for heavy pip dep models)

`pip install --target <dir>` resolves packages completely fresh and ignores the SIF's `/opt/venv`. When torch is a transitive dep, pip downloads a fresh ~6 GB ROCm torch wheel that fills the overlay before other packages install.

**Solution — NFS staging + strip:**
1. Install everything to a staging dir outside the overlay (NFS or any path with ~6 GB free)
2. Strip `torch / torchvision / torchaudio / torchgen / functorch / nvidia / triton` from staging
3. `cp -r` the stripped ~1–2 GB into the overlay

**Why not a pip constraints file:** The venv's torch has a local version string (`torch==2.10.0+rocm7.2.2.gitXXX`). pip cannot find this on any index when resolving `--target` mode, causing `ResolutionImpossible`.

**Why `--extra-index-url` instead:** `--extra-index-url https://download.pytorch.org/whl/rocm7.2` causes pip to pick the ROCm torch wheel, which has **no** `nvidia-*` CUDA co-package deps.

**Prefer `--extra-index-url` + strip over `--no-deps` for runtime installs.** Using `--no-deps` to avoid pulling CUDA torch also silently drops transitive runtime deps (e.g. `tensordict` needs `pyvers` and `treelib`). Instead, use `--extra-index-url` so pip resolves the full dep tree with ROCm torch, then strip `torch / torchvision / torchaudio / nvidia / triton` after install. The ROCm torch download (~3 GB) is wasteful but the install is correct. Reserve `--no-deps` for the overlay build (where overlay size is constrained and dep lists are manually curated).

```bash
# Inside the container during overlay build:
pip install -q --no-cache-dir --target "$STAGE_DIR" \
    --extra-index-url https://download.pytorch.org/whl/${ROCM_WHL_TAG} \
    "<package>" 2>&1 | tail -5

for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
    rm -rf "${STAGE_DIR}/${pkg}" "${STAGE_DIR}/${pkg}"-*.dist-info 2>/dev/null || true
done

cp -r "$STAGE_DIR"/. "$PKG/"   # $PKG = /opt/<model>-pkgs inside overlay
# Safety strip again after cp
for pkg in torch torchvision torchaudio torchgen functorch nvidia triton; do
    rm -rf "${PKG}/${pkg}" "${PKG}/${pkg}"-*.dist-info 2>/dev/null || true
done
```

Always verify at end: `assert 'rocm' in torch.__version__`

**Always build the overlay image on `$TMPDIR` (node-local disk), not on NFS.**

An ext3 overlay image is served via FUSE. Writing thousands of small Python package files through FUSE into an image stored on NFS is extremely slow — both `cp -r` and `tar | tar` are equally bottlenecked because the destination is the FUSE layer over NFS, not the copy tool. The correct pattern:

```bash
LOCAL_OVERLAY="${TMPDIR:-/tmp}/<model>-overlay-${SLURM_JOB_ID:-$$}.img"
apptainer overlay create --size "$OVERLAY_SIZE_MB" "$LOCAL_OVERLAY"

apptainer exec --rocm --overlay "${LOCAL_OVERLAY}:rw" ...  # populate on local disk (fast)

# Copy finished image to NFS — one large sequential write (fast)
cp "$LOCAL_OVERLAY" "$OVERLAY"   # $OVERLAY = NFS destination (set by user, never hardcoded)
rm -f "$LOCAL_OVERLAY"
```

**Validated overlay sizes:**
- ORBIT-2 (pytorch-lightning + xformers + wandb + mpi4py + ...): 7 GB overlay, ~2 GB content
- HydraGNN (torch-scatter + torch-geometric + mpi4py + ...): 4 GB overlay, ~522 MB content
- StormCast (earth2studio + cartopy): 4 GB overlay, ~1.7 GB content

## Local cluster config

Site-specific settings (SLURM partition, account, scratch paths, GPU arch) are stored in an untracked file so they never leak into the public repo. The agent checks two locations (in order):

1. `.cluster-config.yaml` (repo root — gitignored, per-checkout)
2. `~/.config/ai4science-studio/cluster.yaml` (user-level, shared across clones)

**On first run or if neither file exists, run `/init-cluster`.** This command auto-discovers the cluster environment (GPU arch, SLURM partitions/accounts, container runtimes, scratch paths, internet access) and asks the user to confirm via multiple-choice questions. The result is written to one of the above locations.

When cluster-specific info is discovered during a session (new partition, different scratch path, etc.), update the config file so future runs don't re-ask.

**Critical: site-specific paths must never appear in committed scripts — including as env var defaults.** It is not enough that a path comes from an env var; the default value of that env var must also be portable. Any default that contains a username, cluster-specific mount point (`/shared/<user>`, `/scratch/...`), or node name belongs in `.cluster-config.yaml`, not in the script. Use `AI4S_SHARED_DIR` (set by the user) as the base for path defaults, or omit the default entirely and require the user to set the variable.

## Auto-discovery: use `$HOME`, never hardcode `/home`

When searching for SIF files, overlays, or repo clones, always use `$HOME` as the search root — not `/home`. Many HPC clusters mount home directories under non-standard prefixes, so `find /home ...` misses everything. The same applies to overlay and upstream-repo searches.

```bash
find "$HOME" /scratch /projects /opt -maxdepth 4 -name "*.sif" 2>/dev/null | head -20
```

## pip install requires --target (SIF is read-only)

SIF files are squashfs (read-only). All `pip install` inside `apptainer exec` must use `--target <dir>`. Set `PYTHONPATH=<dir>:...`.

**Torch strip is safe** in overlay/pip-packages — the SIF's venv torch is untouched and remains the primary copy.

## No-overlay install-to-tmp pattern

When an sbatch script supports an optional overlay but the user doesn't provide one, deps must be installed into a **host-side temp dir** and **bind-mounted** into the container. The read-only SIF prevents `pip install` into the venv.

```bash
PKGDIR_BIND=()
if [[ -n "$SIF" ]] && { [[ -z "$OVERLAY" ]] || [[ ! -f "$OVERLAY" ]]; }; then
  PKGDIR="${TMPDIR:-/tmp}/<model>-pkgs-${SLURM_JOB_ID:-$$}"
  mkdir -p "$PKGDIR"
  PKGDIR_BIND=(--bind "${PKGDIR}:/opt/<model>-pkgs")

  # Write install script to a host file (avoids quoting issues)
  cat > "$INSTALL_SCRIPT" << 'INSTALLEOF'
  ...pip install --target /opt/<model>-pkgs ...
  INSTALLEOF

  apptainer exec --rocm "${PKGDIR_BIND[@]}" \
      --bind "$(dirname "$INSTALL_SCRIPT"):$(dirname "$INSTALL_SCRIPT")" \
      "$SIF" bash "$INSTALL_SCRIPT"
fi

# Wire PKGDIR_BIND into ALL subsequent apptainer exec calls:
apptainer exec --rocm "${OVERLAY_ARG[@]}" "${PKGDIR_BIND[@]}" ...
```

Key design points:
- `PKGDIR_BIND` is an empty array when overlay IS present → expands to nothing
- The host dir is bound to the same path the overlay would provide (e.g. `/opt/orbit2-pkgs`)
- The rank script's PYTHONPATH stays the same regardless of overlay vs bind-mount
- The install script heredoc uses a **quoted delimiter** (`<< 'EOF'`) so no escaping is needed; pass env vars via `--env`
- **Every** `apptainer exec` in the script must include `"${PKGDIR_BIND[@]}"` — the checkpoint download, the data generation, and the inference run

## Multi-GPU distributed launch (Apptainer + SLURM)

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT="${MODEL_MASTER_PORT:-29500}"

srun --mpi=pmix apptainer exec \
    --rocm \
    "${OVERLAY_ARG[@]}" \
    --env HOSTNAME="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    "$SIF" bash "$RANK_SCRIPT"
```

**Why `--mpi=pmix`:** `mpi4py` auto-calls `MPI_Init` inside the container. Default `srun` and `--mpi=none` both fail because there is no PMIx server reachable through the container namespace. `--mpi=pmix` provides a real PMIx v4 server.

**Why `HOSTNAME` instead of `MASTER_ADDR`:** Several model scripts do `os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]`. Inject rank-0's hostname as `HOSTNAME` so this pattern resolves correctly on all nodes. Using `localhost` only works single-node.

**Multi-node scaling:** Change `--nodes=N --ntasks=N*<gpus_per_node>` in the SBATCH header; the `srun` command is unchanged.

## GPU visibility: `--gpus-per-node` NOT `--gpus-per-task` (MI355X / RCCL)

**Use `--gpus-per-node=8`, NOT `--gpus-per-task=1`** for multi-GPU distributed training with RCCL/NCCL backend on AMD Instinct MI355X.

**Why:** With `--gpus-per-task=1`, SLURM's cgroup restricts each rank to seeing only 1 GPU (`torch.cuda.device_count()=1`). RCCL still tries to read the full KFD topology (`/sys/class/kfd/kfd/topology/nodes/*/io_links/*/properties`) to discover XGMI links, but cannot reconcile 8-GPU topology info with 1-GPU access. This produces:
```
NCCL WARN Could not read node # N
ncclUnhandledCudaError: Call to CUDA function failed.
```

**Fix:** Use `--gpus-per-node=8` (all GPUs visible to all ranks). Frameworks like HydraGNN, DeepSpeed, and plain PyTorch DDP use `SLURM_LOCALID` to select the correct device per rank when `device_count() > 1`:
```python
local_rank = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local_rank)  # rank 0→GPU 0, rank 1→GPU 1, ...
```

**MI355X RCCL env vars (from AMD documentation):**
- Single-node: only `HSA_NO_SCRATCH_RECLAIM=1` needed
- Multi-node: full set (`NCCL_NET_PLUGIN`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, etc.) — see `sbatch_train_amd.sh` in HydraGNN examples

**Multi-node MPI transport (ob1/tcp — correct for Pensando/ionic fabric):**
- Pensando ionic data NICs use `/31` point-to-point subnets; nodes cannot route to each other on data interfaces. Standard IB verbs do NOT work for MPI inter-node traffic.
- MPI uses ob1 PML with TCP BTL over the management NIC: `OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=tcp,self`, `OMPI_MCA_btl_tcp_if_include=<mgmt_iface>`, `MPI4PY_RC_THREADS=false`.
- RCCL uses ANP plugin (`librccl-anp.so`) + `libionic.so.1` (bind-mounted from host) over ionic native transport (RoCEv2/GDRDMA) for GPU allreduce — independent of MPI/UCX.
- Without the ANP bind-mounts, RCCL falls back to socket transport over the management NIC (works but much slower).

## PMIx shared memory fix for Apptainer

When using `srun --mpi=pmix` with Apptainer, ranks may segfault with `PMIx ERROR: UNREACHABLE` if `/dev/shm` or `/tmp` is shared from the host. Fix:
```bash
--env PMIX_MCA_gds=hash
--env PMIX_MCA_psec=native
```

## Never `import` mpi4py-bearing modules from a rank-0-only python before the real `srun`

If a model script does `from mpi4py import MPI` at import time (most do — directly, or transitively via the model's `__init__.py`), the import calls `MPI_Init()` because `mpi4py.rc.initialize=True` is the default. Under `srun --mpi=pmix`, `MPI_Init()` is a **collective rendezvous**: every rank in the SLURM step must arrive at it together before any rank can return.

**Anti-pattern that deadlocks the job:**
```bash
# Inside rank.sh — runs once per SLURM rank, 0..N-1
if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
  python3 - <<'PYEOF'      # short-lived rank-0-only python
    import hydragnn       # ← triggers mpi4py auto-MPI_Init on rank 0 alone
    print("[verify] ...")
PYEOF
fi

exec python -u -c "..."    # the real distributed python on every rank
```

**What goes wrong:** rank 0 enters `MPI_Init()` from the verify python and waits there. Ranks 1..N-1 reach `MPI_Init()` from the main python a moment later. PMIx counts rank 0 as already-arrived for *that* rendezvous, so the verify python returns and exits — taking rank 0's MPI participation with it. When rank 0's main python then calls `MPI_Init()` again, PMIx no longer matches it with ranks 1..N-1, and the whole job hangs in `dist.init_process_group → _create_c10d_store` with no useful error message until the 30-minute TCPStore timeout.

**Symptoms (py-spy on a stuck rank):**
```
_create_c10d_store (torch/distributed/rendezvous.py:200)
_env_rendezvous_handler (torch/distributed/rendezvous.py:281)
init_process_group (torch/distributed/distributed_c10d.py:1806)
setup_ddp (hydragnn/utils/distributed/distributed.py:254)
```
…on every rank except rank 0, which has died or is in a different MPI_Init that will never match.

**Correct patterns (pick one):**

1. **Move the verify inside the main python**, gated on `SLURM_PROCID == 0`. Same information gets logged, but mpi4py's `MPI_Init` happens once per rank, in lockstep:
   ```python
   exec python -u -c "
   import os
   if os.environ.get('SLURM_PROCID', '0') == '0':
       import inspect, hydragnn
       print('[verify]', hydragnn.__file__, flush=True)
   # ...rest of training driver...
   "
   ```

2. **Disable mpi4py auto-init at the top of the verify script** (only safe if the verify never actually calls any MPI op):
   ```python
   import mpi4py
   mpi4py.rc.initialize = False
   import hydragnn  # now safe to import without joining a rendezvous
   ```

3. **Print the verify info on rank 0 from the launching shell only**, never from a python that imports mpi4py-bearing modules.

This bit us once on the HydraGNN perf-analysis recipe. Watch for it whenever you add `python3 -c "..."` scaffolding around an `srun` that uses `--mpi=pmix`.

---

# Docker-specific patterns

## GPU passthrough — two-path detection

Docker needs explicit GPU device injection. Detect AMD Container Toolkit vs raw device passthrough:

```bash
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi "amd"; then
    GPU_ARGS=(--runtime=amd -e AMD_VISIBLE_DEVICES=all)
else
    GPU_ARGS=(--device=/dev/kfd)
    for dev in /dev/dri/renderD*; do
        [[ -e "$dev" ]] && GPU_ARGS+=(--device="$dev")
    done
    GPU_ARGS+=(--group-add video)
fi
```

## Standard Docker run template

```bash
docker run --rm \
    "${GPU_ARGS[@]}" \
    --name "<model>-${SLURM_JOB_ID:-$$}" \
    --network host \
    --shm-size=16g \
    -v "$HOST_DIR":/workspace \
    "$IMAGE" \
    bash -c '
set -euo pipefail
source /opt/venv/bin/activate
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/my/paths:${PYTHONPATH:-}"
...
'
```

**Rules:**
- `--rm` for ephemeral containers (clean exit like Apptainer)
- `--network host` for HuggingFace / PyPI / NOAA downloads and distributed init
- `--shm-size=16g` for PyTorch DataLoader shared memory
- Container name with cleanup trap: `trap 'docker rm -f "$CONTAINER_NAME" 2>/dev/null' EXIT`

## Env vars: append inside container, NEVER via `-e`

`docker run -e LD_LIBRARY_PATH=X` **replaces** the image default entirely. The `rocm/pytorch` image sets `LD_LIBRARY_PATH=/opt/rocm/lib`. Clobbering it drops `libhsa-runtime64.so` and `libamdhip64.so` from the search path, causing `torch.cuda.device_count()=0` and silent CPU fallback.

**Only use `-e` for variables you are introducing** (e.g. `-e ORBIT2_ROOT=/orbit2`, `-e MIOPEN_USER_DB_PATH=/tmp/miopen`). For inherited vars, always append inside the container script.

## Writable container filesystem

Docker containers are writable (unlike Apptainer SIF). Key differences:
- **No `--target` needed** for pip — installs go into the default site-packages
- **No overlay needed** — pip installs persist for the container's lifetime
- **Do NOT strip `torch`** after pip install — it would remove the only copy. Only strip `nvidia`/`triton` CUDA co-packages:
  ```bash
  SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
  for pkg in nvidia triton; do
      rm -rf "${SITE}/${pkg}" "${SITE}/${pkg}"-*.dist-info 2>/dev/null || true
  done
  ```

## MPI in Docker — use torchrun instead

Docker has no `srun --mpi=pmix` equivalent. OpenMPI's `orte_init` needs the full runtime tree which isn't available in most Docker images.

**For multi-GPU distributed models:** Use `torchrun --standalone --nproc_per_node=N` inside a single container with all GPUs passed through. Write a Python rank wrapper that maps torchrun env vars to SLURM env vars if the upstream code expects them:
```python
os.environ["SLURM_NTASKS"]  = os.environ.get("WORLD_SIZE", "8")
os.environ["SLURM_PROCID"]  = os.environ.get("RANK", "0")
os.environ["SLURM_LOCALID"] = os.environ.get("LOCAL_RANK", "0")
```

**For models that import `mpi4py` at module level** without actually using MPI (e.g. `ORBIT_USE_DDSTORE=0`): inject a mpi4py stub package and prepend to `PYTHONPATH`:
```bash
mkdir -p /tmp/mpi4py_stub/mpi4py
python3 -c "
stub = '''
class _RC:
    thread_level = \"serialized\"; threads = False; initialize = False
rc = _RC()
class _CommWorld:
    rank = 0; size = 1
    def allreduce(self, x, op=None): return x
    def Barrier(self): pass
    def bcast(self, obj, root=0): return obj
class MPI:
    COMM_WORLD = _CommWorld()
    SUM = 0; MAX = 1; MIN = 2
'''
open('/tmp/mpi4py_stub/mpi4py/__init__.py', 'w').write(stub)
"
export PYTHONPATH="/tmp/mpi4py_stub:\${PYTHONPATH:-}"
```

**Multi-node Docker is not supported** — use Apptainer + `srun --mpi=pmix` for multi-node distributed workloads.

## apt-get may be blocked

On some clusters, port 80 to Ubuntu mirrors is blocked from containers. Only HTTPS endpoints (PyPI, HuggingFace, NOAA) work. Install everything via `pip`.

## Docker: LD_LIBRARY_PATH must be appended, not set via `-e`

Unlike Apptainer's `--rocm` flag (which injects `/opt/rocm/lib` independently of `LD_LIBRARY_PATH`), Docker has no equivalent mechanism. Passing `docker run -e LD_LIBRARY_PATH=X` replaces the image default entirely, dropping `/opt/rocm/lib` from the search path → `libhsa-runtime64.so` and `libamdhip64.so` unfindable → `torch.cuda.device_count()=0` → silent CPU fallback.

**Always append inside the container script:**
```bash
# Inside the bash -c '...' block, after source /opt/venv/bin/activate:
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

Only use `docker run -e` for variables you are **introducing** (e.g. `-e ORBIT2_ROOT=/orbit2`). Never use it for `LD_LIBRARY_PATH`.

## Performance-analysis subagent workflow (TraceLens + Omnistat user-mode)

This pattern lives in [material_science/models/HydraGNN/recipes/perf-analysis/](../../../material_science/models/HydraGNN/recipes/perf-analysis/) and is documented in detail in [.cursor/skills/ai4science-perf-analysis/SKILL.md](../ai4science-perf-analysis/SKILL.md). Lessons that apply to **any** ai4science model that wants the same workflow:

### 1. ConfigParser does NOT interpolate `os.environ`

Upstream Omnistat configs (`omnistat.ornl`, `omnistat.frontier`) include lines like:
```ini
victoria_datadir = /lustre/.../%(SLURM_JOB_ID)s
```

This **does not work** with stock omnistat-usermode because `omnistat.utils.readConfig` uses plain `configparser.ConfigParser()`, whose `BasicInterpolation` only resolves keys defined inside the config file (DEFAULT section or local section), **not** environment variables. Calling `.get()` on a key with `%(SLURM_JOB_ID)s` raises `InterpolationMissingOptionError`.

**Fix:** Use a `@JOB_DIR@`-style placeholder in the template and run `sed` from the sbatch wrapper at submit time:
```bash
sed -e "s|@JOB_DIR@|${HG_OUTPUT_DIR}|g" omnistat.config.template > $HG_OUTPUT_DIR/omnistat.config
```

### 2. omnistat-usermode `/tmp/omni_rmsjobinfo` permission collision

If the cluster runs system-mode Omnistat as user `omnidc` (or any non-root, non-self user), `/tmp/omni_rmsjobinfo` will already exist and be owned by that user. omnistat-usermode's `omnistat-rms-env` step then fails with `PermissionError` on the very first `--start`, and **exporters never launch** (silent: VictoriaMetrics is up but DB stays empty).

**Fix:** override the path in the user config:
```ini
[omnistat.collectors.rms]
job_detection_mode = file-based
job_detection_file = $AI4S_SHARED_DIR/.../omni_rmsjobinfo
step_detection_file = $AI4S_SHARED_DIR/.../omni_rmsjobinfo_step
```

Path must be on a shared FS reachable by every compute node in the allocation (NFS/Lustre/etc.) because each node's `omnistat-rms-env` runs via `srun -N $NODES --ntasks-per-node=1` and writes to that path.

### 3. VictoriaMetrics needs `-fs.disableMmap` on the login node

A login node may have a high `vm.max_map_count` (e.g. 1048576) but a tight cgroup memory limit (e.g. 8 GB), and even the small (1.6 MB) per-job Omnistat DB cannot mmap successfully — VM panics with:
```
FATAL: cannot mmap "/.../index.bin": cannot mmap file with size 4096 bytes; already memory mapped files: 0: no such device
```

**Fix:** add `-fs.disableMmap` to the VictoriaMetrics command line whenever you start VM on the login node to read a job-scoped DB:
```bash
victoria-metrics-prod -storageDataPath=$DB -fs.disableMmap -httpListenAddr=127.0.0.1:8428 ...
```

### 4. PromQL: filter by jobid via `rmsjob_info` join, not direct label

`rocm_*` metrics from the omnistat collector do **not** carry a `jobid` label themselves. The omnistat-inspect query layer joins them with `rmsjob_info` per-instance:
```promql
avg(rocm_utilization_percentage * on (instance) group_left() (max by (instance) (rmsjob_info{jobid="6762"})))
```

A naive `rocm_utilization_percentage{jobid="6762"}` returns 0 series even though the data is there. Verifier subagents that re-derive metrics from raw curl must use the join pattern.

### 5. HydraGNN `Profile` block goes under `NeuralNetwork`, not the top level

`hydragnn.train.train_validate_test()` is invoked with `config["NeuralNetwork"]` and reads `config["Profile"]` inside that scope:
```python
profiler = Profiler("./logs/" + model_with_config_name)
if "Profile" in config:
    profiler.setup(config["Profile"])
```

So injecting `{"Profile": {"enable": 1, "target_epoch": 1}}` at the top of `gfm_mlip.json` does **nothing** — the profiler stays disabled. The block must be inside `NeuralNetwork`:
```python
cfg.setdefault("NeuralNetwork", {})["Profile"] = {"enable": 1, "target_epoch": epoch}
```

(This is specific to the HydraGNN multi-dataset HPO example; other HydraGNN entrypoints may pass the full config and behave differently. Always trace where the entrypoint reads `Profile` before injecting.)

### 6. Two-phase analyst+verifier pattern

Every claim from the analyst subagent (TraceLens or omnistat-inspect output) is independently re-derived by a verifier subagent from raw data:
- TraceLens claims → re-derived by parsing the raw `.pt.trace.json` event stream.
- Omnistat claims → re-derived via raw `curl` PromQL against the same VictoriaMetrics endpoint.

This catches both tool bugs and analyst hallucinations. Refuted claims stay in `verified_claims.json` (so future iterations don't re-derive the same wrong conclusion); the synthesizer drops only `verdict=refuted` from the final report. In the smoke test, all 6 claims verified within tolerance, but the structure is what catches a future flaky case.

### 7. Datetimes from omnistat-inspect are UTC

`job_info.json` returns `start_time`/`end_time` as ISO-8601 strings without a `Z` or offset — they are **UTC**. Verifier subagents that build PromQL `start`/`end` from these MUST use `calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))`, **not** `time.mktime` (which interprets local time and produces wrong epoch on most clusters). Wrong epoch → empty PromQL result series → false "no data" finding.

### 8. Never read configs from a `/shared` "staged copy" — always from the in-repo template

We hit this once: the omnistat config under `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/omnistat.config` was a stale copy of the in-repo `omnistat.config.template`, with `enable_rocprofiler = False` left over from an early v1 decision. The in-repo template had since been updated to `enable_rocprofiler = True` with the `hbm_flops_f64` profile, but the staged copy was never refreshed, so two perf-analysis runs silently collected zero hardware counters (no HBM bandwidth, no fp64 VALU/MFMA FLOPs). The TraceLens analytical roofline still ran fine, masking the gap until someone asked "where are the HBM counters?"

Two compounding mistakes:
1. The sbatch script defaulted `OMNISTAT_TEMPLATE` to the `/shared` staged path, not the in-repo path.
2. There was no run-time signal that rocprofiler was off — the run banner printed `Omnistat tmpl: <path>` but not the rocprofiler state of that template.

**Fix pattern for any model recipe that reads a config template**:
- Default the path to the in-repo source of truth: `${SCRIPT_DIR}/../recipes/<recipe>/<file>.template`. Only override via env var for ad-hoc probes.
- After resolving the template, parse the critical knobs and **print them in the run banner**, with a `WARN` line if any expected counter group is disabled. Example:
```bash
_ROCPROF_STATE=$(awk -F'[= \t]+' '/^enable_rocprofiler/ {print $2; exit}' "$OMNISTAT_TEMPLATE")
echo "  Rocprofiler  : enable_rocprofiler=${_ROCPROF_STATE:-?}"
[[ "$_ROCPROF_STATE" != "True" ]] && echo "  WARN: rocprofiler counters DISABLED — HBM/FLOP counters NOT collected" >&2
```
- Don't `exit` on it: a config-only probe that intentionally turns rocprofiler off is legitimate. Just make it impossible to be silently wrong.

**Detection rule for any future template**: if a config file lives **both** in `<repo>/.../<recipe>/*.template` and on a shared filesystem (`/shared`, `/lustre`, etc.), assume the shared copy is stale until proven otherwise. Either delete the staged copy or refresh it from the repo before submitting.

### 9. Omnistat rocprofiler-sdk extension is a separate C++ build (not a pip extra)

`pip install -e '.[query]'` does NOT build the rocprofiler-sdk Python binding even with `BUILD_ROCPROFILER_SDK_EXTENSION=1` exported, because pip skips the build step on a re-install of an editable package. Symptom: `enable_rocprofiler = True` in the config + `omnistat-monitor` exporter dies silently with `Missing ROCProfiler-SDK extension: build with installation required` and `ModuleNotFoundError: No module named 'rocprofiler_sdk'`.

The doc-recommended path for editable installs (`docs/installation/extensions.md` upstream) is:
```bash
cd /path/to/omnistat-src
BUILD_ROCPROFILER_SDK_EXTENSION=1 python setup.py build_ext --inplace
```
This must run on a **compute node** (where `/opt/rocm/include/rocprofiler-sdk/` headers and `/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml` are present — login nodes often have neither). Build deps: `cmake-build-extension`, `nanobind`, `cmake>=3.15`, ninja, a C++20 compiler. Output is one `.so` file named `omnistat/rocprofiler_sdk_extension.cpython-3*-x86_64-linux-gnu.so`. Verify with `python3 -c "from omnistat.rocprofiler_sdk_extension import get_samplers, initialize"`.

Imports note: omnistat's collector imports `from omnistat.rocprofiler_sdk_extension import ...`, NOT `import rocprofiler_sdk`. A naive sanity check that imports the upstream `rocprofiler_sdk` Python package will always fail (that package is unrelated, ships with ROCm, and is not on the omnistat venv path).

### 10. gfx950 (MI355X) rocprofiler counter register limits — multiple counters in same hardware block fail

ROCm rocprofiler-sdk on gfx950 returns
```
create profile failed with error code 38: Request exceeds the capabilities of the hardware to collect
```
when a single profile asks for more counters than fit in the per-block register file. Per-block limits are not documented uniformly; we determined empirically on MI355X (gfx950, ROCm 7.2.2):

- **TCC (L2 cache) block**: **`FETCH_SIZE` AND `WRITE_SIZE` together fail.** Either one alone works (with up to 4 SQ_INSTS_VALU_*_F64 counters in the same profile).
- **SQ (scalar/vector unit) block**: ≥4 SQ_INSTS_VALU_*_F64 counters in one profile work.
- Workaround: use `sampling_mode = periodic` and `counters = [[set0], [set1]]` to time-multiplex — but see lesson 11.

Always probe interactively first (1 node `salloc`, run `omnistat-monitor --configfile <test>` foreground for 8 s, grep stderr for "create profile failed") before submitting an sbatch with new counters; the omnistat user-mode launcher swallows the error from the spawned exporter and the only symptom from the outside is "Missing exporter on <host>".

### 11. Omnistat `sampling_mode = periodic` only fires the first set in this build

In `omnistat 1.12.0+git.65ea9ac` (PR #271 + main merged) on MI355X (gfx950) with rocprofiler-sdk freshly built, `sampling_mode = periodic` with two counter sets only fires the FIRST set. Every subsequent set returns identically zero across all cards and timestamps. Half the intended counter coverage is silently lost.

We surfaced this with the `[[FETCH+4×SQ_F64], [WRITE+4×SQ_F64]]` config: FETCH and MFMA_MOPS samples were healthy (~50% non-zero, ~7% non-zero respectively), but WRITE and FMA/ADD/MUL were uniformly zero. Workaround: use `sampling_mode = constant` and a single counter set that fits in the per-block hardware limits — for fp64 + HBM-read on gfx950 that's:
```ini
[omnistat.collectors.rocprofiler.fp64_and_hbm_read]
sampling_mode = constant
counters = ["FETCH_SIZE", "SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_MUL_F64", "SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_MFMA_MOPS_F64"]
```
Trade-off: lose `WRITE_SIZE` (HBM-write bandwidth) until the periodic-mode bug is fixed upstream. For most ML training workloads, FETCH_SIZE is the dominant traffic anyway.

### 12. Three rocprofiler-sdk surfaces — pick the right one

Omnistat exposes **two** independent rocprofiler-sdk integrations, and there's a **third** path that bypasses omnistat entirely. They sample disjoint things and can mostly co-exist; pick by the question you're answering.

| Surface | What you get | Cardinality / cost | Build artifact | Switch in our stack |
|---|---|---|---|---|
| **Device counting** (omnistat collector, `enable_rocprofiler=True`) | Whole-card hardware counters: FETCH_SIZE, fp64 VALU/MFMA, etc. Sampled at 1 Hz, scraped by the omnistat exporter. | One time-series per (card, counter). Constant-mode is essentially free. | `omnistat/rocprofiler_sdk_extension.cpython-*.so` (Python binding, built in-place from `rocprofiler-sdk/CMakeLists.txt` with `BUILD_KERNEL_TRACE_LIB=OFF`) | `enable_rocprofiler = True` in `omnistat.config.template` (default). |
| **Kernel tracing** (omnistat collector, `enable_kernel_trace=True`) | Per-kernel-name dispatch count and total duration on every card, every node. Aggregated in 1 s bins, POSTed via HTTP to the omnistat exporter. | One time-series per (card, kernel-name) — can be 1k+ unique kernel names per training step. Light overhead per dispatch (microseconds), but cardinality blows up VictoriaMetrics if left on for hours. | `build-trace/libomnistat_trace.so` (standalone C++ tool library, built from same CMake with `-DBUILD_KERNEL_TRACE_LIB=ON`) | `OMNISTAT_KERNEL_TRACE=1` env-gate in `sbatch_train_perf_amd.sh` — flips `enable_kernel_trace=True` in the rendered config and exports `ROCP_TOOL_LIBRARIES=<path-to-libomnistat_trace.so>` into the apptainer exec. |
| **`rocprofv3` standalone trace** (no omnistat) | Full kernel + memcpy + HSA API trace on a single rank. JSON / Perfetto / CSV output you read with TraceLens or chrome://tracing. | One trace file per rank, multi-MB per second of training. Heaviest of the three. | None — uses `/opt/rocm/bin/rocprofv3` directly. | Wrap a single rank's command in `rocprofv3 --kernel-trace -o ./rank0_trace -- python ...`. We don't run this in our standard sbatch yet. |

**When to use which:**

- **Device counting only** → multi-node, multi-hour runs. You want HBM bandwidth and fp64 FLOP rate per card across the whole job, with telemetry that survives a 2-hour wallclock without filling the DB.
- **Kernel tracing on top of device counting** → short probe runs (≤ 5 minutes, ≤ 1 node) when you need to know *which kernel* dominated time without rank-0-only kineto traces. Combine with device counting to correlate per-kernel duration against whole-card FLOP rate.
- **`rocprofv3` instead of omnistat** → deepest single-rank kernel detail (per-dispatch timing, register usage, kernel arguments). Use when omnistat's aggregate-by-name view loses the signal you need (e.g. tail latency on one specific dispatch). Don't run it across all ranks; the file size will eat the NFS share.

**Cross-runtime gotcha:** `enable_kernel_trace=True` without the tool library loaded does *nothing* — the omnistat collector just sits with an empty endpoint. The `OMNISTAT_KERNEL_TRACE=1` switch in `sbatch_train_perf_amd.sh` enforces this with a hard `exit 2` if `libomnistat_trace.so` isn't found at the configured path; do **not** loosen that check, otherwise you re-create the exact "silent zero counters" failure mode from §8.

**Build location:** Both omnistat artifacts (the Python extension and the kernel-trace tool library) must be built on a **compute node** inside the same SIF the workload runs in — login nodes don't have apptainer or the ROCm headers, and a host-side build will pick up the wrong libcurl / glibc. `OMNIHUB_TOOLS_DIR` is the shared tools dir from `.cluster-config.yaml` `omnihub.tools_dir` (e.g. `/shared/omnihub/tools`); export it first. Standard build:

```bash
salloc -p <partition> -A <account> -N 1 --time=00:15:00 --gpus-per-node=1 \
  apptainer exec --rocm \
    "${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif" \
    bash -c "cd ${OMNIHUB_TOOLS_DIR}/omnistat-src && \
      cmake -S rocprofiler-sdk/ -B build-trace/ -DBUILD_KERNEL_TRACE_LIB=ON && \
      cmake --build build-trace/ -j 8"
```

Verify with `ls -la ${OMNIHUB_TOOLS_DIR}/omnistat-src/build-trace/libomnistat_trace.so` (~MB-class file, not the tiny 4 kB stub you get when CMake fails halfway).

**SIF build prerequisites (one-time gotchas):** the `pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif` ships **without** `cmake` and **without** libcurl headers (only the `.so.4` runtime). Both omnistat artifacts need cmake; the kernel-trace library additionally needs `<curl/curl.h>`. Working build recipe inside the SIF:

1. `python -m venv /tmp/build_venv && /tmp/build_venv/bin/pip install cmake nanobind zstandard` — installs `cmake` 4.x as a wheel, plus `zstandard` for unpacking modern Ubuntu `.deb` files (which use `data.tar.zst`).
2. Fetch `libcurl4-openssl-dev_*.deb` from `archive.ubuntu.com/ubuntu/pool/main/c/curl/`, `ar x` it, decompress `data.tar.zst` with `zstandard.ZstdDecompressor` in Python, untar to a tmp prefix.
3. Pass `-DCURL_INCLUDE_DIR=<prefix>/usr/include/x86_64-linux-gnu -DCURL_LIBRARY=/usr/lib/x86_64-linux-gnu/libcurl.so.4` to cmake configure. The library finds rocprofiler-sdk via `/opt/rocm/lib/cmake/rocprofiler-sdk/`.

Verified on MI355X (gfx950): 533 kB `libomnistat_trace.so`, ELF64, all rocprofiler-sdk symbols resolved, env vars `OMNISTAT_TRACE_{MAX_INTERVAL,BUFFER_SIZE,ENDPOINT_PORT,LOG}` baked in. `apt-get download` does **not** work in the SIF — apt sources aren't configured — so don't waste a node on it.

**Endpoint port alignment:** the kernel-trace **collector** registers `/kernel_trace` on the same Flask app as the rest of the omnistat exporter, so it listens on `[omnistat.collectors] port` (8101 in our config). The kernel-trace **tool library**'s default `OMNISTAT_TRACE_ENDPOINT_PORT` is 8001, which is wrong for our setup — they must match or every dispatch record gets a connection-refused. The sbatch wrapper auto-derives the port from the rendered config; do not hard-code 8001.

### 13. Empty-string env passthrough breaks `os.getenv() is not None` checks (HydraGNN-class trap)

HydraGNN's `train_validate_test.py` and `preprocess/load_data.py` detect optional knobs with the idiom

```python
if os.getenv("HYDRAGNN_NUM_WORKERS") is not None:
    num_workers = int(os.environ["HYDRAGNN_NUM_WORKERS"])
```

This **fails** when the variable is set to the empty string: `getenv() is not None` returns `True`, and `int("")` raises `ValueError`. Our older `--env KEY="${KEY:-}"` passthrough pattern passes empty strings whenever the wrapper's caller didn't set the var, which masquerades as "set but empty" inside the container.

**Symptom (probe job 7033 on this repo):** all 8 ranks crash on the second line of `create_dataloaders` with `ValueError: invalid literal for int() with base 10: ''` before any kernel ever launches.

**Fix:** build a separate array of `--env` flags that are only appended when the value is non-empty:

```bash
OPT_ENVS=()
for k in HYDRAGNN_NUM_WORKERS HYDRAGNN_PERSISTENT_WORKERS HYDRAGNN_MAX_NUM_BATCH; do
  v="${!k:-}"
  [[ -n "$v" ]] && OPT_ENVS+=( --env "${k}=${v}" )
done

srun ... apptainer exec ... "${OPT_ENVS[@]}" "$HG_SIF" bash "$RANK_SCRIPT"
```

Applied to both `sbatch_train_amd.sh` and `sbatch_train_perf_amd.sh` for HydraGNN. **General lesson** for any model recipe: never use `--env KEY="${KEY:-}"` for variables that the workload reads with `os.getenv(K) is not None` — either pass a numeric default (`--env KEY="${KEY:-0}"`) or use the conditional-array pattern above. Audit every `*/models/*/examples/*.sh` against this when introducing new optional env knobs.

### 14. Cluster-state checks before queueing build/probe jobs

`sinfo -p <partition>` is the cheapest way to know whether a build alloc will ever start. Partitions get drained for slurm maintenance roughly every few weeks — the symptom is `salloc: job NNNN queued and waiting for resources` followed by `(Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions)`. The drain reasons are visible via `sinfo -R`; treat any `slurm deploy maintenance`-class reason as "wait, don't queue". Cancel pending allocs with `scancel` when you discover the drain — they don't time out on their own.

When the cluster is up but you only need a one-shot interactive build inside the SIF, **prefer `srun --no-shell`-equivalent direct exec over `salloc + apptainer exec`**:

```bash
srun -p <partition> -A <account> -N 1 --time=00:15:00 --gpus-per-node=1 --cpus-per-task=8 \
  --job-name=<descriptive> \
  apptainer exec --rocm \
    --bind ${OMNIHUB_TOOLS_DIR}:${OMNIHUB_TOOLS_DIR} \
    "$HG_SIF" \
    bash -c '<your build commands>'
```

Why: `salloc` doesn't auto-bind `${OMNIHUB_TOOLS_DIR}/` into the apptainer namespace. Direct `srun … apptainer exec --bind …` forces the bind explicitly and writes the build artifact to its persistent NFS path in one shot. Verified during the kernel-trace library build (job 7030).

### 15. Dual-failure debug pattern — separate "tool wired correctly?" from "workload happy?"

When a probe job fails, ask **two** questions before retrying with a bigger run:

1. **Did the new tool/instrumentation initialize?** Look for its banner / startup log even on a crashed run — a healthy tool prints something before the workload imports anything heavy.
2. **Did the workload reach the part the tool measures?** A crash at line 863 of `gfm_mlip_all_mpnn.py` (HydraGNN's `create_dataloaders`) is upstream of any GPU kernel launch, so any "0 dispatches captured" is consistent with both "tool broken" and "workload broken before kernels".

**Concrete example from this session (kernel-trace probe job 7033):** all 8 ranks crashed in `int(os.environ["HYDRAGNN_NUM_WORKERS"])` (`ValueError: invalid literal for int() with base 10: ''`). The kernel-trace library *also* printed `[hostname][pid][omnistat] Trace summary: 0/0 processed records (0/0 successful flushes)` per rank — meaning the rocprofiler-sdk tool loaded, registered its callback, and only saw zero dispatches because the workload crashed first. Fixing the env-passthrough bug (§13) and re-running gave 22.5k records/rank — the kernel-trace wiring was correct all along.

**Action:** when you see "0 records / 0 events / 0 metrics" after a failed probe, *don't* re-build the instrumentation. Read the workload error, fix that, and re-run before changing anything tool-side. This saved one full rebuild cycle here.
