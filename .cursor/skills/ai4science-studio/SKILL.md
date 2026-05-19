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
