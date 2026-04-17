---
name: ai4science-studio
description: Applies when working in the AI4Science Studio repository. Describes domain layout, model slug rules, where recipes live, safety expectations, and AMD/ROCm HPC patterns validated in practice.
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
   - `sbatch_<task>_amd.sh` — SLURM batch script; use `--rocm` (not `--nv`) for AMD/Apptainer GPU passthrough. Named `_amd.sh` (not `_mi300x.sh`) — the same script works on MI250X, MI300X, and MI350X with a `rocm7.2.x` image.
   - `build_overlay_amd.sh` — (HPC models with heavy pip deps) builds a persistent Apptainer ext3 overlay pre-loaded with pip deps. Run once per cluster; reuse with `--overlay <path>:ro` to skip 5–15 min pip install on every job. See "Overlay build pattern" below.
   - All scripts must be `chmod +x`.
5. Copy structure from [`_template/`](../../../_template/) when starting a new model folder.
6. For **HPC-oriented** models, consider `recipes/local-cluster-amd.md` (institutional **AMD Instinct** + SLURM/PBS-style notes) and **`data-access.md`** sections on **staging** data (public **Globus** or **Hugging Face CLI** vs copy from **OLCF**/collaborator when public endpoints are not usable).
7. Write recipes around **AMD Instinct** with **PyTorch ROCm** when that matches what upstream documents for supported paths (`gpu_type: "amd"` where configs expose it). Point readers to upstream for install options this repo does not spell out.
8. **GPU arch naming:** Scripts are named `_amd.sh`, not `_mi300x.sh`. The same `rocm7.2.x` image covers MI250X (gfx90a), MI300X (gfx942), and MI350X (gfx950). Only the `ROCM_WHL_TAG` env var and the SIF image need to change for a different ROCm generation — document those as the two knobs, not the GPU model name.

## Canonical AMD/ROCm container image

**Use `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`** for all new models unless upstream requires a specific version.

- Covers MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950)
- Python 3.12, PyTorch 2.10.0+rocm7.2.2
- For older hardware (MI100/gfx908): use a `rocm6.x` image
- For future hardware: update SIF + `ROCM_WHL_TAG` together

Build the SIF once and reuse:
```bash
apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
```

## Standard Apptainer exec template

```bash
apptainer exec \
    --rocm \
    --overlay "${OVERLAY}:ro" \               # omit if no overlay
    --bind "$WORKSPACE":/workspace \
    --env LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib \
    "$SIF" bash /scripts/run.sh
```

**Rules:**
- Always `--rocm`, never manually bind `/opt/rocm` or `/dev/kfd` — `--rocm` handles them and avoids ROCm version mixing
- Always `--env LD_LIBRARY_PATH=...torch/lib` — required for `libcaffe2_nvrtc.so` to be found
- Never bind-mount host `/opt/rocm` when container ROCm > host ROCm (ABI mismatch crashes RCCL)

## Overlay build pattern (for heavy pip dep models)

`pip install --target <dir>` resolves packages completely fresh and ignores the SIF's `/opt/venv`. When torch is a transitive dep, pip downloads a fresh ~6 GB ROCm torch wheel that fills the overlay before other packages install.

**Solution — NFS staging + strip:**
1. Install everything to a staging dir outside the overlay (NFS or any path with ~6 GB free)
2. Strip `torch / torchvision / torchaudio / torchgen / functorch / nvidia / triton` from staging
3. `cp -r` the stripped ~1–2 GB into the overlay

**Why not a pip constraints file:** The venv's torch has a local version string (`torch==2.10.0+rocm7.2.2.gitXXX`). pip cannot find this on any index when resolving `--target` mode, causing `ResolutionImpossible`. Do not use `--constraint torch==<rocm-local-ver>`.

**Why `--extra-index-url` instead:** `--extra-index-url https://download.pytorch.org/whl/rocm7.2` causes pip to pick the ROCm torch wheel, which has **no** `nvidia-*` CUDA co-package deps — the entire CUDA chain is never resolved.

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

## Multi-GPU distributed launch (Apptainer + SLURM)

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT="${MODEL_MASTER_PORT:-29500}"

srun --mpi=pmix apptainer exec \
    --rocm \
    "${OVERLAY_ARG[@]}" \
    --env HOSTNAME="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib \
    "$SIF" bash "$RANK_SCRIPT"
```

**Why `--mpi=pmix`:** `mpi4py` auto-calls `MPI_Init` inside the container. Default `srun` and `--mpi=none` both fail because there is no PMIx server reachable through the container namespace. `--mpi=pmix` provides a real PMIx v4 server.

**Why `HOSTNAME` instead of `MASTER_ADDR`:** Several model scripts do `os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]`. Inject rank-0's hostname as `HOSTNAME` so this pattern resolves correctly on all nodes. Using `localhost` only works single-node.

**Multi-node scaling:** Change `--nodes=N --ntasks=N*<gpus_per_node>` in the SBATCH header; the `srun` command is unchanged.

**MPI workloads must use Apptainer, not Docker:** Docker's `MPI_Init` / `orted` fails. Apptainer shares host namespaces and picks up the host MPI runtime transparently.

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

Delete `pip-packages/` or rebuild the overlay entirely when switching to a different base SIF image. Packages compiled for a different Python version or NumPy ABI will crash silently (e.g. "do not import numpy from its source directory"). Stale `.dist-info` also confuses pip resolution.
