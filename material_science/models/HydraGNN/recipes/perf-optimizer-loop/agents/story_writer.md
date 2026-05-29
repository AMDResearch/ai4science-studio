# story_writer subagent

Produces the final progression narrative + a matplotlib FOM-vs-iteration chart
once the loop has completed (or aborted). Reads only files; writes only
`story.md` and `foms.png` under the loop directory.

## Inputs

- `loop-<uuid>/foms.csv` — every iteration's row, in chronological order
- `loop-<uuid>/STATUS.txt` — full event log
- `loop-<uuid>/do_not_retry.json` — rejected lever ids
- `loop-<uuid>/iter-*-lever.json` — per-iter chosen lever (the `reason` field is the lever_picker's justification)
- `<jobid>/combined_report.md` for every iter that completed (resolved via symlinks `loop-<uuid>/iter-N-<lever>.json`)
- `<jobid>/foms.json` for every iter that completed
- `<jobid>/kernel_correlation.csv` for every iter that has it (best-iter likely; may be missing if fom_extractor returned partial)
- [`../lever_catalog.yaml`](../lever_catalog.yaml) — for citation URLs and lever metadata
- [`../README.md`](../README.md) — for the references list to mirror at the end of story.md

## Outputs

- `loop-<uuid>/story.md` — the progression narrative
- `loop-<uuid>/foms.png` — 2-panel matplotlib chart
- Final stdout: `STATUS=ok; reason=story written, n_iters=<N>, best=iter-<i>=<lever>` or `STATUS=partial; reason=<missing inputs>`

## `story.md` layout

```markdown
# HydraGNN iterative-sysopt-loop — <uuid>

**Hardware:** 2 nodes × 8 GPUs MI355X (gfx950).
**Image:** `pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`.
**Loop:** <N> iterations, started <ISO timestamp>, ended <ISO timestamp>, total wall <H>h<M>m.
**Primary FOM:** `epoch_time_s` (mean over epoch >= 1, drops warm-up).

## TL;DR

| Iter | Lever | epoch_time_s | Δ vs prev best | Accepted? | Notes |
|---|---|---|---|---|---|
| 0 | baseline | <x> | n/a | yes | |
| 1 | batch_800 | <x> | <pct> | <y/n> | |
| ... | | | | | |

Best result: **iter-<i> `<lever>`** at `<x>` s/epoch (<pct>% vs baseline; <pct>% vs R2 from `recipes/perf-analysis/`).
<one-sentence why this lever helped>.

## Initial state of the system (iter-0 baseline)

<3-4 sentences: what the analysts found about the baseline. Cite numbers from
iter-0/combined_report.md — top bottleneck class, mean util, MFMA TFLOP/s,
exposed NCCL share, dominant kernels. End with a sentence naming the loop's
working hypothesis (e.g. "still dispatch-bound, expect batch_800 to lift it").>

## Iteration-by-iteration progression

For each iter (i from 1 to N):

### Iter <i>: `<lever>` <ACCEPTED/REJECTED [reason]>

**Why this lever was chosen:** <copy lever_picker reason verbatim from iter-N-lever.json>.

**What changed:** env-var diff or rank-script patch from lever_catalog.yaml.

**Result (vs previous best):**
- epoch_time_s: <prev> → <new> (Δ <pct>%)
- throughput_samples_per_s: <prev> → <new> (Δ <pct>%)
- mfma_tflops: <prev> → <new>
- energy_J / mean_power_W / energy_per_sample_J: as above
- final_loss: <prev> → <new> (control)

**Bottleneck shift (from <jobid>/combined_report.md):**
<2-3 sentences. What was the #1 bottleneck before vs after?>

**Verdict:** ACCEPTED (current best updated) / REJECTED (auto-reverted; added to do_not_retry.json).

**Citation:** [<blog/doc>](<url>) — from lever_catalog.yaml.

---

## Final state of the system

<3-4 sentences: where is the workload now (top bottleneck, MFMA % of peak, mean util, energy per sample). Compare to the initial state and the historical R2 (`recipes/perf-analysis/`) baseline. End with what would be the next-best lever IF we extended the catalog.>

## FOM progression chart

![FOM progression across iterations](foms.png)

Top panel: `epoch_time_s` (primary, lower is better) and `energy_per_sample_J` (lower is better, twin axis).
Bottom panel: `throughput_samples_per_s` and `mfma_tflops` (both higher is better).
Rejected iterations marked with red x. Diagnostic iterations marked with grey diamond.

## Levers rejected (auto-revert)

| Iter | Lever | epoch_time_s | regression % | Likely reason |
|---|---|---|---|---|
| ... | ... | ... | ... | <copy from combined_report.md if it explains; else "unknown — see iter-N combined_report.md"> |

## Levers in catalog not tried

| Lever | Why not tried |
|---|---|
| <id> | <"loop ended before its turn" / "in do_not_retry from a prior loop" / "diagnostic-only, gate not met"> |

## What would be next (outside this loop)

<2-3 sentences naming the most promising direction not covered by the current catalog.>

## References

<copy the README.md References section verbatim, scoped to those whose URLs appear in the chosen-levers' citation field.>

---
Generated <ISO timestamp> by [`material_science/models/HydraGNN/recipes/perf-optimizer-loop/agents/story_writer.md`](.).
```

## `foms.png` layout (matplotlib)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top panel: epoch_time_s + energy_per_sample_J on twin axis
ax_top.set_xlabel("Iteration")
ax_top.set_ylabel("epoch_time_s (lower = better)", color="C0")
ax_top.plot(iters_x, epoch_times, "C0-o", label="epoch_time_s")
ax_top_t = ax_top.twinx()
ax_top_t.set_ylabel("energy_per_sample_J (lower = better)", color="C1")
ax_top_t.plot(iters_x, energy_per_sample, "C1-s", label="energy_per_sample_J")

# Mark rejected with red x at the epoch_time location
for x, et, acc in zip(iters_x, epoch_times, accepted_flags):
    if acc == "false":
        ax_top.plot(x, et, "rx", markersize=14, markeredgewidth=3)
    elif acc == "diagnostic":
        ax_top.plot(x, et, "D", color="grey", markersize=10)

# Bottom panel: throughput + MFMA TFLOP/s
ax_bot.set_xlabel("Iteration")
ax_bot.set_ylabel("throughput_samples_per_s (higher = better)", color="C2")
ax_bot.plot(iters_x, throughput, "C2-o", label="throughput")
ax_bot_t = ax_bot.twinx()
ax_bot_t.set_ylabel("mfma_tflops per card (higher = better)", color="C3")
ax_bot_t.plot(iters_x, mfma, "C3-s", label="mfma_tflops")

# Annotate each point with the lever id
for ax in (ax_top, ax_bot):
    for x, lever in zip(iters_x, lever_ids):
        ax.annotate(lever, (x, ax.get_ylim()[1]), xytext=(0, -12),
                    textcoords='offset points', rotation=30,
                    fontsize=7, ha='center')

fig.suptitle(f"HydraGNN sysopt loop {uuid} — {n_iters} iters")
fig.tight_layout()
fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
```

Use `matplotlib.use("Agg")` for headless rendering. Do NOT use seaborn (extra dep not in the omnistat venv). Use the omnistat venv at `${OMNIHUB_TOOLS_DIR}/omnihub-inspect/bin/python` which already has matplotlib via TraceLens deps.

## Style rules

- Quote specific FOM numbers; never use "much better" / "a lot".
- Cite the lever_picker's `reason` verbatim — that's the story of WHY each lever was chosen.
- For ACCEPTED iters, the delta is against the iter that was the best before this iter ran. For REJECTED iters, the delta is against current_best (which doesn't change on rejection).
- If `kernel_trace_diag_only` fired, name what was learned from it ("step 3's kernel-trace pinned the dominant kernel to `<name>`, which led iter 4 to pick `<lever>`").
- If the loop hit early-stop (5% convergence rule), say so in TL;DR; if it exhausted the catalog, say so; if STOP flag was hit, say so with timestamp.

## Failure modes

| Failure | Action |
|---|---|
| foms.csv empty / missing | STATUS=fail; reason=no_foms |
| only iter-0 ran | STATUS=partial; story written with just baseline section + the lever picker's analysis of what would have been next |
| any per-iter combined_report.md missing | substitute "(combined_report.md missing for this iter; see <jobid>/hydragnn-train-*.out)" in the bottleneck-shift section |
| matplotlib unavailable | STATUS=partial; story.md still written; foms.png omitted with a banner at the top of story.md |

## Notes for the implementing agent

- Read at most the files listed in `## Inputs`.
- Atomic write: tmpfile + rename for both story.md and foms.png.
- The story is auto-generated each time; it MUST be self-contained (someone reading just story.md should understand the run without opening any other file). Cross-link the per-iter combined_report.md files but never assume the reader will follow them.
