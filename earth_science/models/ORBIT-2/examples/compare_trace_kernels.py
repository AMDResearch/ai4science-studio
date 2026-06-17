#!/usr/bin/env python3
"""Compare GPU-kernel device-time breakdown between two Kineto traces.

Tool-independent (no TraceLens): aggregates `traceEvents` with cat=="kernel" by name, summing `dur`,
and rolls them into the categories that matter for the ORBIT-2 conv bottleneck (GEMM/Cijk, im2col,
naive_conv, Winograd, attention, elementwise/other). Shares (% of total kernel time) are the
comparable quantity across runs because the captured step windows differ.

Usage:
  compare_trace_kernels.py LABEL_A=/path/a.pt.trace.json LABEL_B=/path/b.pt.trace.json [--md OUT.md]
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

CATEGORIES = [
    ("GEMM (Cijk/hipBLASLt)", re.compile(r"Cijk|gemm|Gemm|hipblaslt|rocblas", re.I)),
    ("im2col lowering", re.compile(r"im2col|Im2Col|Im2d2Col", re.I)),
    ("naive_conv (MIOpen direct)", re.compile(r"naive_conv", re.I)),
    ("Winograd conv", re.compile(r"winograd", re.I)),
    ("implicit-GEMM conv (MIOpen)", re.compile(r"miopen.*(igemm|implicit)|implicit.*gemm", re.I)),
    ("attention/SDPA", re.compile(r"attention|flash|sdpa|fmha|attn|bwd_kernel_(dq|dk|dv|fuse)|fwd_kernel", re.I)),
    ("layernorm/norm", re.compile(r"layer_norm|layernorm|GroupNorm|cuComputePartGrad|cuComputeGradInput", re.I)),
    ("elementwise/copy/reduce", re.compile(r"elementwise|vectorized|copy|reduce|cat_|fill|index", re.I)),
    ("collective/comm (RCCL)", re.compile(r"nccl|rccl|AllReduce|ReduceScatter|AllGather|all_gather|reduce_scatter", re.I)),
]


def load_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        data = json.load(fh)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def kernel_times(path):
    """Return (by_name: {name: total_us}, total_us)."""
    by_name = defaultdict(float)
    for ev in load_events(path):
        if ev.get("cat") == "kernel" and "dur" in ev:
            by_name[ev.get("name", "?")] += float(ev["dur"])
    return by_name, sum(by_name.values())


def categorize(by_name):
    cat_tot = defaultdict(float)
    for name, us in by_name.items():
        for label, rx in CATEGORIES:
            if rx.search(name):
                cat_tot[label] += us
                break
        else:
            cat_tot["other"] += us
    return cat_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="LABEL=/path/to/trace.pt.trace.json")
    ap.add_argument("--md", default=None)
    ap.add_argument("--topn", type=int, default=12)
    args = ap.parse_args()

    runs = []
    for p in args.pairs:
        label, _, path = p.partition("=")
        by_name, tot = kernel_times(path)
        runs.append((label, path, by_name, tot, categorize(by_name)))

    lines = ["# MIOpen ORNL-flags validation — GPU kernel device-time breakdown", ""]
    for label, path, _bn, tot, _cat in runs:
        lines.append(f"- **{label}**: total GPU-kernel time = {tot/1000:.1f} ms  (`{path}`)")
    lines.append("")

    # category share table
    cats = [c for c, _ in CATEGORIES] + ["other"]
    header = "| Category | " + " | ".join(f"{lbl} %" for lbl, *_ in runs) + " |"
    sep = "|---|" + "---|" * len(runs)
    lines += ["## Category share (% of total GPU-kernel time)", "", header, sep]
    for c in cats:
        cells = []
        for *_x, tot, cat in runs:
            share = 100.0 * cat.get(c, 0.0) / tot if tot else 0.0
            cells.append(f"{share:.1f}")
        if any(float(x) > 0.05 for x in cells):
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")

    # top kernels per run
    for label, _path, by_name, tot, _cat in runs:
        lines += [f"## Top {args.topn} kernels — {label}", ""]
        lines += ["| kernel | ms | % |", "|---|---|---|"]
        for name, us in sorted(by_name.items(), key=lambda kv: -kv[1])[: args.topn]:
            lines.append(f"| `{name[:80]}` | {us/1000:.1f} | {100*us/tot if tot else 0:.1f} |")
        lines.append("")

    out = "\n".join(lines)
    print(out)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(out)
        print(f"\n[written] {args.md}")


if __name__ == "__main__":
    main()
