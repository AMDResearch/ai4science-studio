# Run HydraGNN perf-analysis or perf-optimizer-loop on an AMD cluster

Guide the user through **performance engineering** for HydraGNN Predictive GFM: one-shot **perf-analysis** (2-node training + TraceLens + Omnistat diagnosis) or iterative **perf-optimizer-loop** (accept/revert on **`epoch_time_s`**). This is **not** ensemble inference — use `/run-hydragnn` for that.

**Recipe docs:** [material_science/models/HydraGNN/recipes/perf-analysis/README.md](../material_science/models/HydraGNN/recipes/perf-analysis/README.md), [material_science/models/HydraGNN/recipes/perf-optimizer-loop/README.md](../material_science/models/HydraGNN/recipes/perf-optimizer-loop/README.md).

**Attribution / levers:** [dispatch-attribution.md](../material_science/models/HydraGNN/recipes/perf-optimizer-loop/dispatch-attribution.md), [lever_catalog.yaml](../material_science/models/HydraGNN/recipes/perf-optimizer-loop/lever_catalog.yaml).

**Orchestration reference:** `.cursor/skills/ai4science-perf-analysis/SKILL.md` (launcher → parallel analysts → parallel verifiers → synthesizer).

## Step 0 — Cluster config check (required)

Read `.cluster-config.yaml` (repo root) or `~/.config/ai4science-studio/cluster.yaml`.

**Required for perf recipes:**

- Set **`AI4S_SHARED_DIR`** from cluster config (`paths.scratch` or documented scratch root).
- Set **`PERF_TOOLS_DIR`** from `perf_tools.dir`. If missing, stop and send the user to **`/init-cluster`** Q9 (Performance tooling).

Pre-fill SLURM partition/account from config.

**Runtime:** Apptainer + SLURM only (`sbatch_train_perf_amd.sh`).

---

## Step 1 — Questionnaire (ask ALL before acting)

**Q0. Mode**

- **perf-analysis** — One 2-node perf job + post-hoc multi-subagent analysis (default topology for this recipe).
- **perf-optimizer-loop** — Iterative tuning; primary FOM **`epoch_time_s`** (lower is better); see `recipes/perf-optimizer-loop/README.md` for stop rules and `do_not_retry.json`.

**Q1. HydraGNN checkout (`HG_REPO_DIR`)**

Default in sbatch: `$AI4S_SHARED_DIR/models/HydraGNN/code/HydraGNN` (cloned at pinned SHA if missing).

- **Use default** / **Yes, custom path** / **No** (clone instructions) / **Auto-discover** — `find "$HOME" /scratch /projects /opt -maxdepth 5 -type d -name "HydraGNN" 2>/dev/null | head -20` and confirm `examples/multidataset_hpo_sc26` exists.

**Q2. SIF (`HG_SIF`)**

Default: `$AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif`.

- **Yes** / **No** / **Auto-discover** — same SIF discovery pattern as `/run-orbit2`.

**Q3. Overlay (`HG_OVERLAY`)**

Default: `$AI4S_SHARED_DIR/models/HydraGNN/overlays/hydragnn-overlay.img`.

- **Yes** / **No, build one** — `sbatch material_science/models/HydraGNN/examples/build_overlay_amd.sh` (edit `#SBATCH` partition/account first) / **Auto-discover** — `find ... -name "*hydragnn*overlay*"`.

**Q4. Datasets (`HG_DATASETS`, `HG_DATA_DIR`)**

Default datasets and paths are set in `sbatch_train_perf_amd.sh` (`ANI1x,Alexandria` under `HG_DATA_DIR`). Confirm `.bp` files exist or user will stage per [recipes/data/README.md](../material_science/models/HydraGNN/recipes/data/README.md).

**Q5. Precision**

Perf recipe defaults to **`fp64`** / **`HG_PRECISION=fp64`** (MACE-style equivariance). **`bf16` is not used** for this GFM path — do not suggest bf16 as a lever; the lever catalog documents blocked options.

**Q6. Blocked or high-risk levers (set expectations)**

Point users to `lever_catalog.yaml` entries marked **`blocked`** or **`accepted-baked-in`**, including at minimum:

- **`torch.compile`** on HydraGNN MLIP MACE — blocked (double-backward / AOT autograd).
- **`torch.jit.script`** — blocked (keyword-arg expansion at MACE interaction block; see lever catalog).
- **TunableOp live or warmup-then-use** — blocked on this stack (documented GPU faults during tuning); do not re-propose without a stack update and catalog change.

**Q7. Partition and account**

- **Provide manually** / **Auto-discover** — `sinfo` + `sacctmgr` as in `/run-orbit2`.

**Q8. (perf-optimizer-loop only) Iteration budget**

Default **5**; run inside **`tmux`**. Optional: `export ANTHROPIC_API_KEY=...` for Claude Code CLI orchestration inside `run_optimizer_loop.sh`.

---

## Step 2 — Auto-discovery

Reuse SIF/overlay/SLURM snippets from `.claude/commands/run-orbit2.md`.

**HydraGNN repo:**

```bash
find "$HOME" /scratch /projects /opt -maxdepth 5 -type d -name "HydraGNN" 2>/dev/null | head -20
```

Confirm branch **`Predictive_GFM_2024`** if user cares about upstream alignment (see model README).

---

## Step 3 — Run

```bash
export AI4S_SHARED_DIR=<from cluster config>
export PERF_TOOLS_DIR=<from cluster config perf_tools.dir>
# Optional overrides:
# export HG_SIF=...  HG_OVERLAY=...  HG_REPO_DIR=...  HG_DATASETS=...  HG_DATA_DIR=...
```

Edit **`#SBATCH`** in **`material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh`** for partition/account.

**Default perf-analysis job is 2-node** — ensure `#SBATCH --nodes=2` (or user override) matches intent.

### perf-analysis

```bash
sbatch material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh
```

After completion, drive subagents per **`ai4science-perf-analysis` SKILL** using prompts under `material_science/models/HydraGNN/recipes/perf-analysis/agents/`.

### perf-optimizer-loop

Preflight only (smoke; no Claude invocation):

```bash
bash material_science/models/HydraGNN/examples/run_optimizer_loop.sh "$(uuidgen)" 5 --preflight-only
```

Full loop:

```bash
tmux new -s hg-perf-loop
bash material_science/models/HydraGNN/examples/run_optimizer_loop.sh "$(uuidgen)" 5
```

Graceful stop: `touch "$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/loop-<uuid>/STOP"`.

---

## Step 4 — Monitor and artifacts

```bash
squeue -j <job_id>
tail -f "$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/<job_id>/"*.out
```

**perf-analysis:** `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/<jobid>/` — `combined_report.md`, `manifest.json`, `tracelens/`, `omnistat/`, traces under `logs/`.

**perf-optimizer-loop:** `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/loop-<uuid>/` — `STATUS.txt`, `foms.csv`, `story.md`, per-iteration job subdirs.

---

## Arguments

$ARGUMENTS
