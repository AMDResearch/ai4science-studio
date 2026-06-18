#!/usr/bin/env python3
"""Build baseline_report.{md,json} for one-node GPU saturation documentation.

Reads manifest.json, rendered training YAML (path from manifest), SLURM log,
and foms.json (runs FOM extraction if missing). Extracts batch_size, data_type,
model preset and key widths from YAML via lightweight regex (no PyYAML).

Parameter count: searches the log for common Lightning / summary patterns;
if absent, leaves placeholders for manual fill (see one-node-gpu-baseline.md).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# YAML snippets (avoid PyYAML dependency on login nodes)
# -----------------------------------------------------------------------------

_RE_BATCH = re.compile(r"^\s*batch_size:\s*(\d+)\s*$", re.MULTILINE)
_RE_DTYPE = re.compile(r"^\s*data_type:\s*(\S+)\s*$", re.MULTILINE)
_RE_PRESET = re.compile(r"^\s*preset:\s*(\S+)\s*$", re.MULTILINE)
_RE_EMBED = re.compile(r"^\s*embed_dim:\s*(\d+)\s*$", re.MULTILINE)
_RE_DEPTH = re.compile(r"^\s*depth:\s*(\d+)\s*$", re.MULTILINE)
_RE_DEC = re.compile(r"^\s*decoder_depth:\s*(\d+)\s*$", re.MULTILINE)
_RE_FSDP = re.compile(r"^\s*fsdp:\s*(\d+)\s*$", re.MULTILINE)
_RE_SDDP = re.compile(r"^\s*simple_ddp:\s*(\d+)\s*$", re.MULTILINE)


def _first_int(rx: re.Pattern[str], text: str) -> int | None:
    m = rx.search(text)
    return int(m.group(1)) if m else None


def _first_str(rx: re.Pattern[str], text: str) -> str | None:
    m = rx.search(text)
    return m.group(1).strip().strip('"').strip("'") if m else None


def _bytes_per_param(data_type: str | None) -> float:
    if not data_type:
        return 4.0
    dt = data_type.lower()
    if dt == "bfloat16":
        return 2.0
    if dt in ("float16", "half"):
        return 2.0
    if dt == "float64":
        return 8.0
    return 4.0  # float32 default


# Log lines: Lightning / PyTorch / ORBIT sometimes print these on rank 0
_PARAM_RES = [
    re.compile(r"Total\s+model\s+parameters:\s*([\d.]+)\s*([KkMmGg])\b", re.IGNORECASE),
    re.compile(r"(\d[\d,]*)\s+Total params", re.IGNORECASE),
    re.compile(r"Total\s+params:\s*(\d[\d,\s]*)", re.IGNORECASE),
    re.compile(r"trainable\s+params:\s*(\d[\d,\s]*)", re.IGNORECASE),
    re.compile(r"Model\s+parameters?:\s*(\d[\d,\s]*)", re.IGNORECASE),
    re.compile(r"num[_\s]?param(?:eter)?s?:\s*(\d[\d,\s]*)", re.IGNORECASE),
]


def _parse_int_human(s: str) -> int:
    return int(s.replace(",", "").replace(" ", "").strip())


def _extract_params_from_log(log_text: str) -> int | None:
    for line in log_text.splitlines():
        for rx in _PARAM_RES:
            m = rx.search(line)
            if not m:
                continue
            try:
                if m.lastindex and m.lastindex >= 2 and m.group(2):
                    val = float(m.group(1))
                    suf = m.group(2).upper()
                    mult = {"K": 1e3, "M": 1e6, "G": 1e9}.get(suf, 1.0)
                    return int(val * mult)
                return _parse_int_human(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _infer_ai4s_shared_from_job_dir(job_dir: Path) -> Path | None:
    """.../<AI4S_SHARED_DIR>/models/ORBIT-2/perf-runs/<jobid> -> <AI4S_SHARED_DIR>."""
    try:
        parts = job_dir.resolve().parts
        if "models" in parts and "ORBIT-2" in parts:
            i = parts.index("models")
            if i >= 2:
                return Path(parts[0]).joinpath(*parts[1:i])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _resolve_slurm_log(job_dir: Path, manifest: dict) -> Path | None:
    job_id = str(manifest.get("job_id") or job_dir.name)
    candidates: list[Path] = []
    mlog = manifest.get("slurm_log")
    if mlog:
        candidates.append(Path(mlog))
    candidates.append(job_dir / f"orbit2-train-{job_id}.out")
    shared = os.environ.get("AI4S_SHARED_DIR")
    if not shared:
        inferred = _infer_ai4s_shared_from_job_dir(job_dir)
        if inferred is not None:
            shared = str(inferred)
    if shared:
        candidates.append(
            Path(shared) / "models" / "ORBIT-2" / "outputs" / "train" / "logs" / f"orbit2-train-{job_id}.out"
        )
    candidates.append(Path.cwd() / f"orbit2-train-{job_id}.out")
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def _resolve_rendered_yaml(job_dir: Path, manifest: dict) -> Path | None:
    rc = manifest.get("rendered_config")
    if rc:
        p = Path(rc)
        if p.is_file():
            return p.resolve()
    job_id = str(manifest.get("job_id") or job_dir.name)
    for name in (f"interm_8m_prism_{job_id}.yaml", f"interm_8m_era5_{job_id}.yaml"):
        p = job_dir / name
        if p.is_file():
            return p.resolve()
    for p in sorted(job_dir.glob("interm_*.yaml")):
        if p.is_file() and ("prism" in p.name or "era5" in p.name):
            return p.resolve()
    return None


def _minimal_manifest(job_dir: Path) -> dict:
    jid = job_dir.name
    return {
        "job_id": jid,
        "job_dir": str(job_dir),
        "slurm_log": str(job_dir / f"orbit2-train-{jid}.out"),
    }


def _ensure_foms(job_dir: Path, log_path: Path) -> Path:
    out = job_dir / "foms.json"
    if out.is_file():
        return out
    fom_script = Path(__file__).resolve().parent / "run_fom_extractor.py"
    if not fom_script.is_file():
        return out
    rc = subprocess.call(
        [sys.executable, str(fom_script), "--job-dir", str(job_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if rc != 0 and not out.is_file():
        sys.stderr.write(f"warning: run_fom_extractor.py exited {rc}; foms.json may be missing\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="ORBIT-2 one-node GPU baseline report.")
    ap.add_argument("--job-dir", type=Path, required=True, help="perf-runs/<jobid>/")
    ap.add_argument("--num-params", type=int, default=None, help="Override parameter count if known")
    args = ap.parse_args()
    job_dir: Path = args.job_dir.resolve()
    if not job_dir.is_dir():
        print(f"error: job dir not found: {job_dir}", file=sys.stderr)
        return 2

    manifest_path = job_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _minimal_manifest(job_dir)
        sys.stderr.write(f"warning: no manifest.json; using job dir name as job_id={manifest['job_id']}\n")

    cfg_path = _resolve_rendered_yaml(job_dir, manifest)
    if cfg_path is None:
        print(f"error: could not find rendered training YAML under {job_dir}", file=sys.stderr)
        return 2
    yaml_text = cfg_path.read_text(encoding="utf-8", errors="replace")

    batch_size = _first_int(_RE_BATCH, yaml_text)
    data_type = _first_str(_RE_DTYPE, yaml_text)
    preset = _first_str(_RE_PRESET, yaml_text)
    embed_dim = _first_int(_RE_EMBED, yaml_text)
    depth = _first_int(_RE_DEPTH, yaml_text)
    dec_depth = _first_int(_RE_DEC, yaml_text)
    fsdp = _first_int(_RE_FSDP, yaml_text)
    simple_ddp = _first_int(_RE_SDDP, yaml_text)

    par = manifest.get("parallelism") or {}
    fsdp = fsdp or par.get("fsdp")
    simple_ddp = simple_ddp or par.get("simple_ddp")

    log_path = _resolve_slurm_log(job_dir, manifest)
    if log_path is None:
        print(f"error: SLURM log not found for job {manifest.get('job_id', job_dir.name)}", file=sys.stderr)
        return 2
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    _ensure_foms(job_dir, log_path)
    foms: dict = {}
    foms_path = job_dir / "foms.json"
    if foms_path.is_file():
        foms = json.loads(foms_path.read_text(encoding="utf-8"))

    n_params = args.num_params
    if n_params is None:
        n_params = _extract_params_from_log(log_text)

    bpp = _bytes_per_param(data_type)
    param_bytes = int(n_params * bpp) if n_params is not None else None

    worlds = (fsdp or 0) * (simple_ddp or 0)
    global_batch = batch_size * worlds if batch_size and worlds else None

    report: dict = {
        "job_dir": str(job_dir),
        "job_id": manifest.get("job_id"),
        "orbit2_root": manifest.get("orbit2_root"),
        "git_sha": manifest.get("git_sha"),
        "git_branch": manifest.get("git_branch"),
        "rendered_config": str(cfg_path),
        "trainer": {
            "batch_size_per_rank": batch_size,
            "data_type": data_type,
            "global_batch_approx": global_batch,
        },
        "parallelism": {"fsdp": fsdp, "simple_ddp": simple_ddp, "world_size": worlds},
        "model_yaml": {
            "preset": preset,
            "embed_dim": embed_dim,
            "depth": depth,
            "decoder_depth": dec_depth,
        },
        "model_size": {
            "num_parameters": n_params,
            "param_bytes_if_dense_weights": param_bytes,
            "bytes_per_param_assumption": bpp,
            "note": "Activations/optimizer/grads not included; use Omnistat VRAM + TraceLens for live footprint.",
        },
        "foms": foms,
        "artifacts": {
            "slurm_log": str(log_path),
            "traces_dir": manifest.get("trace_dir"),
            "omnistat_db": manifest.get("omnistat_db"),
        },
    }

    (job_dir / "baseline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# ORBIT-2 GPU baseline — job `{manifest.get('job_id')}`",
        "",
        "## Config",
        "",
        f"- **Rendered YAML:** `{cfg_path}`",
        f"- **ORBIT2_ROOT:** `{manifest.get('orbit2_root')}`",
        f"- **Git:** `{manifest.get('git_sha', '?')}` (`{manifest.get('git_branch', '?')}`)",
        "",
        "## Parallelism / batching",
        "",
        f"| fsdp | simple_ddp | world | per-rank batch | global batch (approx.) |",
        f"|------|------------|-------|----------------|-------------------------|",
        f"| {fsdp or '—'} | {simple_ddp or '—'} | {worlds or '—'} | {batch_size or '—'} | {global_batch or '—'} |",
        "",
        "## Model (YAML fingerprint)",
        "",
        f"| preset | embed_dim | depth | decoder_depth |",
        f"|--------|-----------|-------|---------------|",
        f"| {preset or '—'} | {embed_dim or '—'} | {depth or '—'} | {dec_depth or '—'} |",
        "",
        "## Model size (parameters)",
        "",
    ]
    if n_params is not None:
        md_lines.extend(
            [
                f"- **Parameter count (from log or --num-params):** `{n_params:,}`",
                f"- **Dense weight bytes (params × {bpp} B, {data_type or 'float32'}):** ~{param_bytes:,} bytes (~{param_bytes / 1e9:.3f} GB)",
            ]
        )
    else:
        md_lines.extend(
            [
                "- **Parameter count:** *not found in log* — paste from a Lightning summary or run the container one-liner in "
                "[`one-node-gpu-baseline.md`](../recipes/perf-analysis/one-node-gpu-baseline.md) §4, then re-run:",
                "",
                "```bash",
                f"python3 examples/report_orbit2_gpu_baseline.py --job-dir {job_dir} --num-params <N>",
                "```",
            ]
        )

    md_lines.extend(
        [
            "",
            "## Figures of merit (ORBIT log)",
            "",
        ]
    )
    if foms:
        md_lines.append("```json")
        md_lines.append(json.dumps({k: foms[k] for k in ("steady_batch_time_s", "loss_sanity_pass", "steady_batch_count") if k in foms}, indent=2))
        md_lines.append("```")
    else:
        md_lines.append("*Run `run_fom_extractor.py` — no foms.json yet.*")

    md_lines.extend(
        [
            "",
            "## Utilization evidence (fill after analysis)",
            "",
            "- **Omnistat:** attach `omnistat/report_summary.md` + key PromQL plots (VRAM, `FETCH_SIZE` / MFMA-derived FLOPs).",
            "- **TraceLens:** attach `tracelens/report_summary.md` from rank-0 trace.",
            "- **Saturation checklist:** higher `ORBIT2_BATCH_SIZE` until VRAM plateaus **and** `steady_batch_time_s` is acceptable vs batch sweep.",
            "",
        ]
    )

    (job_dir / "baseline_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {job_dir / 'baseline_report.md'}")
    print(f"Wrote {job_dir / 'baseline_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
