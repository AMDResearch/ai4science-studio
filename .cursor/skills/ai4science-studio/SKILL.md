---
name: ai4science-studio
description: Applies when working in the AI4Science Studio repository. Describes domain layout, model slug rules, where recipes live, safety expectations, and AMD/ROCm HPC patterns (Apptainer and Docker) validated in practice.
---

# AI4Science Studio (repository)

## Repository map

- **Domains:** `earth_science/` (includes climate and weather), `material_science/`, `protein_folding/`, `healthcare/`, `physics_simulation/`.
- **Models:** `<domain>/models/<model-slug>/` with a `README.md` per model and `recipes/` for that model only.
- **Index:** Root [`README.md`](../../../README.md).

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
5. Copy structure from [`_template/`](../../../_template/) when starting a new model folder.
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

**Validated overlay sizes:**
- ORBIT-2 (pytorch-lightning + xformers + wandb + mpi4py + ...): 7 GB overlay, ~2 GB content
- StormCast (earth2studio + cartopy): 4 GB overlay, ~1.7 GB content

## pip install requires --target (SIF is read-only)

SIF files are squashfs (read-only). All `pip install` inside `apptainer exec` must use `--target /workspace/pip-packages`. Set `PYTHONPATH=/workspace/pip-packages:...`.

**Torch strip is safe** in overlay/pip-packages — the SIF's venv torch is untouched and remains the primary copy.

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
