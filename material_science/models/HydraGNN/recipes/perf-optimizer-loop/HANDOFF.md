# HydraGNN perf-optimizer-loop — handoff to the next agent

> **Read this first.** Single-file orientation for any agent picking up this work mid-flight. Keyword-rich title so a glob/grep for `HANDOFF`, `next agent`, `perf-optimizer`, `MI355X`, `dispatch-bound`, `HydraGNN handoff` will surface it.

**Last touched:** 2026-05-26 by aaji.
**Branch:** `aaji/perf-optimizer-loop-hydragnn` (6 local commits ahead of origin; NOT pushed yet).
**Ticket:** [DCSWEAP-4502](https://amd.atlassian.net/browse/DCSWEAP-4502) "Run HydraGNN on MI355X". Latest comment id `22139024` (posted 2026-05-26 by aaji) is the current state of record.

## TL;DR — where the work is

The 2-node MI355X performance characterization on Vultr Lux is **done**. Current best for HydraGNN GFM-MLIP MACE on 2 nodes (16× MI355X) is **75.0 s/epoch** with this exact env (already the sbatch default):

```bash
HG_BATCH_SIZE=400
HG_PRECISION=fp64
HYDRAGNN_NUM_WORKERS=8
HYDRAGNN_PERSISTENT_WORKERS=1
HG_NUM_EPOCH=3
HYDRAGNN_MAX_NUM_BATCH=50
PROFILE_TARGET_EPOCH=2
TORCH_NCCL_HIGH_PRIORITY=1     # the only accepted lever across 3 loops
GPU_MAX_HW_QUEUES=2            # ditto
```

The workload is **dispatch-bound** (HIP launch latency ceiling 3.7 µs × 270 K launches/s, identical on every MI355X we probed). fp32-vs-fp64 differs by < 2 % despite fp32 MFMA having 2× the peak, which is the canonical proof. The path to a step-change requires **two upstream HydraGNN PRs** (see "Next work" below).

## Where every artefact lives

| What | Where | Tracked? |
|---|---|---|
| Recipe + agent prompts | `material_science/models/HydraGNN/recipes/perf-optimizer-loop/` | git |
| Lever catalog (machine-readable; blocked levers have evidence + path-to-unblock) | `recipes/perf-optimizer-loop/lever_catalog.yaml` | git |
| Sbatch wrapper (defaults = current best config) | `material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh` | git |
| Optimizer-loop driver (Claude Code CLI, tmux) | `material_science/models/HydraGNN/examples/run_optimizer_loop.sh` | git |
| Reusable node-health microbench (mount + STREAM + HIP launch + GPU/firmware inventory; ~30 s) | `material_science/models/HydraGNN/examples/microbench_node_health.sh` | git |
| All lessons learned (perf-analysis-specific) | `.cursor/skills/ai4science-perf-analysis/SKILL.md` (lessons 1-19) | git |
| Per-loop runtime artefacts (3 loops + 2 investigations) | `/shared/aaji/models/HydraGNN/perf-runs/loop-*` and `/shared/aaji/models/HydraGNN/perf-runs/{compile-investigation-*,pathc-pathb-*}` | NOT tracked |
| Microbench raw outputs + REPORT.md + SYSADMIN_MESSAGE.md | `/shared/aaji/microbench-a-vs-b/` | NOT tracked |
| Container (PyTorch 2.10.0 + ROCm 7.2.2 + HydraGNN overlay) | `/shared/aaji/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif` | NOT tracked |
| Top-level ticket | [DCSWEAP-4502](https://amd.atlassian.net/browse/DCSWEAP-4502) | external |

## Reproduce the current best in one command

```bash
ssh aaji@rad-vultr-login           # or whoever the next user is
cd /home/aaji/git/ai4science-studio
git checkout aaji/perf-optimizer-loop-hydragnn
export AI4S_SHARED_DIR=/shared/aaji
sbatch material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh
# expect: 75.0 s/epoch, train_validate_test ≈ 230 s, J/sample ≈ 1.631
```

## Next work (priority order, from the ticket)

| # | Task | Estimate | Notes |
|---|---|---|---|
| 1 | **8-node + 16-node strong-scaling sweep** with current best config | 1-2 hours wall, mostly queue time | Cluster cap is 16 nodes. Modify `sbatch_train_perf_amd.sh` `--nodes=N` only. Capture epoch time + J/sample + scaling efficiency vs the 2-node baseline. Write into a new `recipes/perf-optimizer-loop/scaling-results/` markdown + foms.csv. |
| 2 | **Open the two HydraGNN upstream PRs** | a few days each | (a) Move `energy_force_loss` into `EnhancedModelWrapper.forward` (~30-50 LOC across `hydragnn/models/create.py:622-758` + `hydragnn/train/train_validate_test.py:725-735`) — unblocks `torch.compile`. (b) Unpack `**conv_args` at `hydragnn/models/MACEStack.py:397` — unblocks `torch.jit.script`. Reference issues: see `lever_catalog.yaml` entries `torch_compile_e3nn` and `torch_jit_script` for the full evidence + diff sketches. Cite `/shared/aaji/models/HydraGNN/perf-runs/compile-investigation-aa480554-*/INVESTIGATION_REPORT.md` and `/shared/aaji/models/HydraGNN/perf-runs/pathc-pathb-a3b8f3f0-*/pathc-result.md` in the PRs. |
| 3 | **Second-user reproducibility** | half day | Have a different user run the one-command repro above on a different login session. If they hit 75 ± 1 s/epoch, ticket acceptance #7 is satisfied. |
| 4 | **Frontier-scale path documentation (DDStore phase 2)** | document only | Cannot validate on Vultr Lux (cap=16); write up the 504-node path in `recipes/perf-optimizer-loop/scaling-results/frontier-path.md`. Reference DDStore upstream docs. |

## Landmines to NOT step on (these all cost real hours to discover)

Each item is cross-referenced to a lesson in `.cursor/skills/ai4science-perf-analysis/SKILL.md` for the full backstory.

| Don't | Why | Lesson |
|---|---|---|
| Don't re-try `torch.compile` on HydraGNN MLIP without first landing PR #2a above | AOT-Autograd doesn't support double backward; `energy_force_loss` calls `autograd.grad(create_graph=True)`. Tested 6 hooks across 3 hypotheses, all fail identically. | #6, #7 |
| Don't re-try `torch.jit.script` without first landing PR #2b above | TorchScript JIT can't expand `**conv_args` at `MACEStack.py:397`. | #14 |
| Don't enable `PYTORCH_TUNABLEOP_TUNING=1` in any form (live, warmup, warmup-then-use) | All variants fault all 16 GPUs in the hipBLASLt tuning routine. Re-test only after a documented ROCm/hipBLASLt fix. | #8, #15 |
| Don't draw a-nodes-vs-b-nodes conclusions from FOM deltas | They're physically identical (verified 2026-05-26); the historical "17 % gap" was three baselines on the same b1+b2 pair. Always `jq .nodes_list iter-0-baseline.json` first. | #10, #18 |
| Don't pick `num_workers_12` based on the loop-43b33ec1 win | Same lever regressed in loop-c3e4df1c on the same node pair; it's noise > effect at this measurement granularity. | #9, #18 |
| Don't pick `batch_800` for HydraGNN MLIP MACE | Dispatch saturates at batch=400; +87 % at batch=800. The R2 recommendation was from a pre-MLIP HydraGNN config. | lever_catalog `batch_800` notes |
| Don't pick `precision_fp32` expecting a speedup | The workload is dispatch-bound; fp32 gives < 2 % difference. (It IS still useful as a one-shot dispatch-vs-compute discriminator on new workloads — see lesson #13.) | #13 |
| Don't pick `nccl_minchannels=112` at N=2 | +10.3 % at N=2; only meaningful at ≥8 nodes. | lever_catalog `nccl_minchannels` notes |
| Don't omit `AI4S_SHARED_DIR` from `sbatch --export=ALL,...` | `--export=ALL` doesn't include unexported shell vars; sbatch dies on line 61 with `AI4S_SHARED_DIR must be set`. | #12 |
| Don't use `--gpus-per-node=N` for srun probes | SLURM filter rejects with "Invalid GRES specification"; use `--gres=gpu:amd_instinct_mi355_oam:N` instead. | (sbatch wrapper) |
| Don't claim a class-wide node bandwidth gap from a 16-GB memory cgroup | `cpu_mem_bw.sh`-style measurements get an allocation-artefact ceiling at ~220 GB/s; the real hardware delivers ~313 GB/s when allocated full-node. The committed `microbench_node_health.sh` works under either allocation but the threshold is calibrated to 200 GB/s for the cgroup case. | (microbench REPORT) |
| Don't blacklist a node *class* on a single failure | Probe per-node with the committed mount-health probe (exit 42) or `microbench_node_health.sh` and exclude only the named bad node(s). | #1, #18 |

## Per-node health to be aware of (sysadmin will resolve)

These are observations to relay to the cluster admin (draft is at `/shared/aaji/microbench-a-vs-b/SYSADMIN_MESSAGE.md`), NOT ticket items. If you land on one of these nodes, exclude by name and continue:

| Node | Symptom | Workaround |
|---|---|---|
| `lux-mi355x-a5` | Intermittent dual-NUMA STREAM degradation (179 vs 220 GB/s COPY, 8 % jitter) | `--exclude=lux-mi355x-a5` |
| `lux-mi355x-a6` | `/home` + `/shared` bind-mount EIO inside Apptainer (historical 2026-05-22) | Mount probe (exit 42) auto-detects; orchestrator retries with `--exclude` |
| `lux-mi355x-a10` | PMIx ring collective times out at MPI init (historical 2026-05-23) | `--exclude=lux-mi355x-a10` |

## Bootstrap checklist for a brand-new agent

1. **Read [`README.md`](README.md)** in this directory for the recipe topology + iteration shape.
2. **Read [`lever_catalog.yaml`](lever_catalog.yaml)** in this directory — entries marked `status: blocked` or `status: accepted-baked-in` matter most.
3. **Read** `.cursor/skills/ai4science-perf-analysis/SKILL.md` lessons #1-19, especially the **REVISED** markers on lessons 9, 10, 16.
4. **`git log --oneline -8`** on this branch to see the change history with one-liners.
5. **`cat /shared/aaji/models/HydraGNN/perf-runs/loop-*/foms.csv`** to see the actual FOM trajectories across iterations.
6. **Check ticket** [DCSWEAP-4502](https://amd.atlassian.net/browse/DCSWEAP-4502), especially the latest comment (id `22139024`, posted 2026-05-26).
7. **Pick one of "Next work" items above** and start.

## Open questions / decisions for the human, NOT for the agent

- Should the branch be pushed to `origin` and a draft PR opened against `main`? (User hasn't asked; agent should not push without explicit ask.)
- Should the two HydraGNN upstream PRs be opened by the agent or by a human? (They're against `github.com/ORNL/HydraGNN`, a third-party repo; agent can prepare patches but the human should be the author of record.)
- Is there appetite for adding `microbench_node_health.sh` to other models' recipe folders (StormCast, ORBIT-2, GP-MoLFormer)? The script is generic enough to apply to any MI355X workload on Lux.

## Authors / contacts

- **Performance work + this branch:** Ashwin Aji (`ashwin.aji@amd.com`, github: TBD)
- **DCSWEAP-4502 reporter:** Nicholas Malaya (`nicholas.malaya@amd.com`)
- **Upstream HydraGNN maintainers:** ORNL — see [`github.com/ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) issues for the right tag

## If anything in this file goes stale

Update this file in the same commit as the change that made it stale. Single-source-of-truth for handoff state — if it disagrees with reality, fix the file, don't write a "HANDOFF-v2.md".
