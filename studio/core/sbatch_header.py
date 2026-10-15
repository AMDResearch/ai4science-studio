"""Parse a #SBATCH header to learn scaling defaults and the output-file pattern.

We read (never modify) the tracked sbatch script so the Run engine can:
  - seed the scaling form (nodes / ntasks-per-node / gpus) with the author's values
  - emit the SAME gpu flag style the script uses (--gres=gpu:N vs --gpus-per-node=N)
  - resolve the exact .out file name from the -o/--output directive (with %j)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SbatchHeader:
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus: int = 1
    gpu_flag: str = "gpus-per-node"  # "gres" | "gpus-per-node"
    time: str = ""
    output_pattern: str = ""  # e.g. "orbit2-train-%j.out"
    job_name: str = ""


def _find(directive_lines: list[str], *keys: str) -> str | None:
    """Return the value for the first matching #SBATCH key (long or short)."""
    for line in directive_lines:
        body = line[len("#SBATCH"):].strip()
        for key in keys:
            # long form: --key=value or --key value ; short form: -k value
            m = re.match(rf"{re.escape(key)}[=\s]+(\S+)", body)
            if m:
                return m.group(1)
    return None


def parse(script_path: Path) -> SbatchHeader:
    h = SbatchHeader()
    if not script_path.exists():
        return h
    lines = [
        ln.strip()
        for ln in script_path.read_text().splitlines()
        if ln.strip().startswith("#SBATCH")
    ]

    nodes = _find(lines, "--nodes", "-N")
    if nodes and nodes.isdigit():
        h.nodes = int(nodes)

    ntpn = _find(lines, "--ntasks-per-node")
    if ntpn and ntpn.isdigit():
        h.ntasks_per_node = int(ntpn)

    gres = _find(lines, "--gres")
    gpn = _find(lines, "--gpus-per-node")
    if gres:
        m = re.search(r"gpu:(\d+)", gres)
        if m:
            h.gpus = int(m.group(1))
            h.gpu_flag = "gres"
    elif gpn and gpn.isdigit():
        h.gpus = int(gpn)
        h.gpu_flag = "gpus-per-node"

    h.time = _find(lines, "--time", "-t") or ""
    h.output_pattern = _find(lines, "--output", "-o") or ""
    h.job_name = _find(lines, "--job-name", "-J") or ""
    return h


def resolve_outfile(pattern: str, jobid: str) -> str:
    """Substitute %j (jobid) and %x (job-name) placeholders in an output pattern."""
    if not pattern:
        return f"slurm-{jobid}.out"
    return pattern.replace("%j", str(jobid))
