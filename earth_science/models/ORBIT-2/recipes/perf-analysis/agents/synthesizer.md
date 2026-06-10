# synthesizer subagent

Merge `tracelens/verified_claims.json` and `omnistat/verified_claims.json` into a single ranked, deduplicated bottleneck report.

## Inputs

- `<perf_run_dir>/manifest.json`
- `<perf_run_dir>/tracelens/verified_claims.json`
- `<perf_run_dir>/omnistat/verified_claims.json`
- (Optional) `<perf_run_dir>/tracelens/report_summary.md`, `<perf_run_dir>/omnistat/report_summary.md`

## Outputs

- `<perf_run_dir>/combined_report.md`

## Steps

### 1. Load and filter

```python
import json
tl = json.load(open(f"{perf_run}/tracelens/verified_claims.json"))
om = json.load(open(f"{perf_run}/omnistat/verified_claims.json"))
all_claims = [{"src":"TL", **c} for c in tl] + [{"src":"OS", **c} for c in om]
```

Drop `verdict == "refuted"`. Keep `inconclusive` claims but tag them so the user sees the gap.

### 2. Deduplicate by class + topic

Group claims by `class`. Within each group, merge claims that look like the same finding from two sources. Heuristic:
- Both have `class=comm_scaleout` → almost certainly the same finding viewed from trace (NCCL kernels) and telemetry (network rates). Merge into one entry that lists both `src` values and both magnitudes.
- One says `gpu_compute` low TFLOP/s, the other says `gpu_memory_hbm` high HBM% — these are likely the **same** root cause (memory-bound kernel) seen two ways. The synthesizer should call this out as a single finding with both signatures.

### 3. Rank

Score = `magnitude.value × confidence_weight × (corroborated ? 1.5 : 1.0)`.
`confidence_weight`: high=1.0, medium=0.7, low=0.4. `corroborated` = both TL and OS contributed to the merged entry.

### 4. Tag "system limit reached"

A claim is a system limit if any of:
- `proposed_remedy is null`
- both `verdict=verified` and `remedy_probe.delta_pct` is small (< 5%)
- the metric matches a documented MI355X spec ceiling (e.g. fp64 39 TFLOP/s, HBM 8 TB/s, ANP scale-out ~25 GB/s)

Mark these explicitly so the report doesn't promise a fix that won't materialize.

### 5. Write `combined_report.md`

Layout:

```markdown
# ORBIT-2 bottleneck analysis — JOBID <id>

**Run:** 2 nodes × 8 GPUs (MI355X), <runtime>s, precision=<p>, batch=<b>
**Sources:** TraceLens v<x> on rank-0 trace, Omnistat user-mode (<sampling>s sampling)
**Verdict counts:** <verified>/<refuted>/<inconclusive>

## Executive summary

<1 paragraph, 4-5 sentences. Named bottleneck class first, magnitude, then "system-limit" or "improvable">

## Ranked bottlenecks

### 1. <Class> — <pithy title>  [score: <s>] [<corroborated|TL-only|OS-only>] [<system-limit|improvable>]

**Hypothesis:** <merged hypothesis>
**Magnitude:** <number, sources>
**Evidence:** <bullets, with paths>
**Remedy:** <if any> — **tried?** <yes/no>; **delta:** <if tried>
**Verifier note:** <verifier_evidence digest>

### 2. ...

## Remedies tried in this session

| Class | Probe | Baseline | After remedy | Delta |
|---|---|---|---|---|
| ... |

## Remedies proposed but not tried (cost/risk)

| Class | Remedy | Why deferred |
|---|---|---|

## Limits reached

| Class | Spec ceiling | Observed |
|---|---|---|

## Inconclusive claims

| Source | Class | Hypothesis | Why inconclusive |
|---|---|---|---|

## Next steps

1. <most informative A/B run, e.g. fp64 vs bf16 at 1 node>
2. <next observability gap to fill, e.g. enable rocprofiler hbm counters>
3. <next iteration: multi-rank trace fusion / rank-N comparison / longer run>

## Artifacts

- `tracelens/report.xlsx`
- `tracelens/verified_claims.json`
- `omnistat/inspect_outputs/`
- `omnistat/verified_claims.json`
- `manifest.json`

---
Generated <ISO timestamp> by `material_science/models/ORBIT-2/recipes/perf-analysis/`.
```

### 6. Final stdout

```
STATUS=ok; reason=<n_findings>; top=<class>; tried=<n_probes>
```

## Style rules

- Specific numbers from evidence — never vague ("most of the time", "huge").
- If TL and OS disagree on a number that's not refuted, show **both** with sources rather than pick one.
- When `remedy_probe.ran=false` because of multi-node requirement, say so explicitly.
- Do not editorialize about ORBIT-2 as a model; only about the run.

## Failure modes

- Both inputs missing → STATUS=fail.
- Only one input present → emit a partial report with a banner explaining which side is missing; STATUS=partial.
