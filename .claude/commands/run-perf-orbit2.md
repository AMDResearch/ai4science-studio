# Run ORBIT-2 perf-analysis or perf-optimizer-loop on an AMD cluster

Guide the user through **performance engineering** workflows for ORBIT-2 (Bayes-CAST EDM / same-dir timing configs): one-shot **perf-analysis** (TraceLens + Omnistat diagnosis) or iterative **perf-optimizer-loop** (accept/revert on throughput). This is **not** inference/visualization — use `/run-orbit2` for that.

**Recipe docs:** [earth_science/models/ORBIT-2/recipes/perf-analysis/README.md](../earth_science/models/ORBIT-2/recipes/perf-analysis/README.md), [earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/README.md](../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/README.md).

**Orchestration reference (Cursor / any agent):** read `.cursor/skills/ai4science-perf-analysis/SKILL.md` for the launcher → analyst → verifier → synthesizer loop.

## Step 0 — Cluster config check (required)

Read `.cluster-config.yaml` (repo root) or `~/.config/ai4science-studio/cluster.yaml`.

**Required for perf recipes:**

- `paths.scratch` (or equivalent) → use as **`AI4S_SHARED_DIR`** (same convention as other studio scripts).
- `omnihub.tools_dir` → use as **`OMNIHUB_TOOLS_DIR`** (layout: `omnihub-inspect/`, `omnistat-src/`, `victoriametrics/`). If missing or empty, stop and tell the user to re-run **`/init-cluster`** and answer **Q9 (Performance tooling)**.

Pre-fill SLURM partition/account from `slurm.partition` / `slurm.account` when present.

**Runtime note:** Perf workflows here assume **Apptainer + SLURM** (same as `sbatch_train_perf_amd.sh`). There is no Docker perf path in the recipes.

---

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the following. Do not assume defaults. Wait for answers before proceeding.

**Q0. Mode**

- **perf-analysis** — Submit one training perf job; after it finishes, drive subagents per `ai4science-perf-analysis` SKILL (`agents/launcher.md`, analysts, verifiers, `synthesizer.md`) to produce `combined_report.md`.
- **perf-optimizer-loop** — Iterative loop: one lever per iteration, accept/revert on **`throughput_samples_per_s`** with **`loss_sanity_pass`** as a control (see perf-optimizer-loop README).

**Q1. Code root (`ORBIT2_ROOT`)**

Bayes-CAST checkout used for training (must contain `launch/train_edm.py` for the default perf path).

- **Yes** — user provides full path.
- **No** — generate clone instructions (user may use Bayes-CAST fork or path documented in perf-analysis README).
- **Auto-discover** — search for a directory containing `launch/train_edm.py`.

**Q2. SIF**

Apptainer image for ROCm PyTorch `rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` (or user override).

- **Yes** / **No** (pull command) / **Auto-discover** — same pattern as `/run-orbit2`.

**Q3. ORBIT-2 overlay**

Pre-built `orbit2-overlay.img` (recommended).

- **Yes** / **No, build one** (`sbatch earth_science/models/ORBIT-2/examples/build_overlay_amd.sh`) / **Auto-discover** — same pattern as `/run-orbit2`.

**Q4. Data**

- **`ORBIT2_DATA_ROOT`** — Staged ERA5 NPZ tree (perf defaults use Bayes-CAST template + ERA5; see perf-analysis README).
- **`ORBIT2_CONFIG_TEMPLATE`** — Optional; default **`edm_8m_era5_1x8.yaml`** in perf docs. For HBM saturation / staging notes: [STAGING_ERA5_FOR_HBM.md](../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md).
- **`interm_8m_prism.yaml` / `interm_8m_era5.yaml`** — Public ORBIT-2 same-dir configs (timing-only); document if used instead of Bayes-CAST EDM.

**Q5. Topology**

- **1 node × 8 GPUs** (default for phase-1 loop and perf-analysis quick start).
- **Multi-node** — e.g. `sbatch --nodes=2 ...`; RCCL / GID / `LD_LIBRARY_PATH` caveats in perf-analysis README and `sbatch_train_perf_amd.sh`. After one-node loop converges, see perf-optimizer-loop README **Two-node gate**; optional driver: `earth_science/models/ORBIT-2/examples/run_2node_scaleout_loop.sh`.

**Q6. Partition and account**

- **Provide manually** / **Auto-discover** (same as `/run-orbit2`).

**Q7. (perf-optimizer-loop only) Iteration budget**

Max iterations (default **5**). Confirm use of **`tmux`** on the login node so SSH disconnect does not kill the driver.

**Q8. (perf-optimizer-loop only) Claude Code CLI driver**

`run_optimizer_loop.sh` can invoke the orchestrator via Claude Code CLI when **`ANTHROPIC_API_KEY`** is set. If the user has no API key, explain they can still run **preflight-only** steps and manual `sbatch` per iteration using `recipes/perf-optimizer-loop/agents/orchestrator.md` as a checklist.

---

## Step 2 — Auto-discovery (when chosen)

**Code root containing `launch/train_edm.py`:**

```bash
find "$HOME" /scratch /projects /opt -maxdepth 6 -type f -path "*/launch/train_edm.py" 2>/dev/null | head -20
```

Derive `ORBIT2_ROOT` as the parent of `launch/`.

**SIF / overlay / SLURM:** reuse the same `find` / `sinfo` / `sacctmgr` snippets as in `.claude/commands/run-orbit2.md` (SIF: `*.sif` filtered for rocm/pytorch; overlay: `*orbit2*overlay*`).

Always confirm discovered paths with the user before `sbatch`.

---

## Step 3 — Run

Export from cluster config + user answers:

```bash
export AI4S_SHARED_DIR=<from cluster config paths.scratch>
export OMNIHUB_TOOLS_DIR=<from cluster config omnihub.tools_dir>
export ORBIT2_ROOT=<path to Bayes-CAST or ORBIT-2 tree with launch/train_edm.py>
export ORBIT2_SIF=<path>                    # if not using default under $AI4S_SHARED_DIR/images/...
export ORBIT2_OVERLAY=<path>                # optional if script defaults suffice
export ORBIT2_DATA_ROOT=<staged era5 root>
# export ORBIT2_CONFIG_TEMPLATE=edm_8m_era5_1x8.yaml   # optional
```

Edit **`#SBATCH`** lines in **`earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh`** for partition/account (and `--nodes` if multi-node).

### perf-analysis

```bash
# From ai4science-studio repo root
sbatch earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh
```

After the job completes, follow **`.cursor/skills/ai4science-perf-analysis/SKILL.md`**: dispatch launcher (if not already folded into job), `tracelens_analyst` + `omnistat_analyst` in parallel, verifiers in parallel, then `synthesizer`. Prompts live under `earth_science/models/ORBIT-2/recipes/perf-analysis/agents/`.

### perf-optimizer-loop

Optional repo file check (does not validate cluster env):

```bash
bash earth_science/models/ORBIT-2/examples/validate_orbit2_optimizer_loop_recipe.sh
```

Preflight only (recommended first):

```bash
bash earth_science/models/ORBIT-2/examples/run_optimizer_loop.sh "$(uuidgen)" 5 --preflight-only
```

Full loop (inside **`tmux`**):

```bash
tmux new -s orbit2-perf-loop
bash earth_science/models/ORBIT-2/examples/run_optimizer_loop.sh "$(uuidgen)" 5
# detach: Ctrl-b d
```

Graceful stop: `touch "$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/loop-<uuid>/STOP"`.

---

## Step 4 — Monitor and artifacts

**SLURM job:**

```bash
squeue -j <job_id>
tail -f "$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<job_id>/"*.out
```

**perf-analysis deliverables** (under `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/`): `manifest.json`, `omnistat-db/`, `tracelens/`, `combined_report.md`, `foms.json` (when extractor run), traces under `traces/`.

**perf-optimizer-loop** (under `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/loop-<uuid>/`): `STATUS.txt`, `foms.csv`, `story.md`, `do_not_retry.json`, `known_bad_nodes.txt`; per-iter job dirs mirror perf-analysis layout.

**Primary FOM (loop):** `throughput_samples_per_s` (higher is better); keep **`loss_sanity_pass`**. GEMM / lever evidence: [gemm-attribution.md](../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/gemm-attribution.md), [lever_catalog.yaml](../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/lever_catalog.yaml).

---

## Arguments

$ARGUMENTS
