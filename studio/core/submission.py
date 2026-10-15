"""Build the environment + sbatch command for a run, and submit it.

Principles:
  - Never modify tracked files. Partition/account/scaling go in as sbatch CLI
    flags (they override the in-file #SBATCH directives). Model params go in as
    exported environment variables.
  - Omit unset optionals entirely; never emit VAR="" (some scripts do int(VAR)).
  - cwd = the model's examples/ dir (relative -o outputs and SCRIPT_DIR rely on it).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import sbatch_header
from .catalog import Model, Recipe


@dataclass
class SubmitSpec:
    partition: str = ""
    account: str = ""
    qos: str = ""
    nodes: int = 1
    ntasks_per_node: int = 8
    gpus: int = 8
    gpu_flag: str = "gpus-per-node"  # or "gres"
    time: str = ""
    env: dict[str, str] = field(default_factory=dict)  # only the vars to set


@dataclass
class BuiltCommand:
    argv: list[str]
    cwd: str
    env_delta: dict[str, str]
    outfile_pattern: str
    script_path: str


def build_command(model: Model, recipe: Recipe, spec: SubmitSpec) -> BuiltCommand:
    if not recipe.sbatch_script:
        raise ValueError(f"Recipe {recipe.task} has no sbatch script")

    script_path = model.abs_path / recipe.sbatch_script
    cwd = model.examples_dir
    header = sbatch_header.parse(script_path)

    argv: list[str] = ["sbatch"]
    if spec.partition:
        argv.append(f"--partition={spec.partition}")
    if spec.account:
        argv.append(f"--account={spec.account}")
    if spec.qos:
        argv.append(f"--qos={spec.qos}")
    argv.append(f"--nodes={spec.nodes}")
    argv.append(f"--ntasks-per-node={spec.ntasks_per_node}")

    # Match the gpu-flag style the script's own header uses.
    flag = spec.gpu_flag or header.gpu_flag
    if flag == "gres":
        argv.append(f"--gres=gpu:{spec.gpus}")
    else:
        argv.append(f"--gpus-per-node={spec.gpus}")

    if spec.time:
        argv.append(f"--time={spec.time}")

    argv.append(str(script_path))

    return BuiltCommand(
        argv=argv,
        cwd=str(cwd),
        env_delta=dict(spec.env),
        outfile_pattern=header.output_pattern,
        script_path=str(script_path),
    )


def render_preview(built: BuiltCommand) -> str:
    """Human-readable dry-run: exported env + the sbatch command line."""
    lines = [f"cd {built.cwd}"]
    for k, v in built.env_delta.items():
        lines.append(f"export {k}={_shquote(v)}")
    lines.append(" \\\n  ".join(built.argv))
    return "\n".join(lines)


def _shquote(v: str) -> str:
    if v and re.fullmatch(r"[A-Za-z0-9_./:=-]+", v):
        return v
    return "'" + v.replace("'", "'\\''") + "'"


@dataclass
class SubmitResult:
    ok: bool
    jobid: str = ""
    stdout: str = ""
    stderr: str = ""
    outfile: str = ""


def can_submit() -> bool:
    return shutil.which("sbatch") is not None


def submit(built: BuiltCommand) -> SubmitResult:
    """Submit via bare sbatch with a merged environment (default --export=ALL)."""
    if not can_submit():
        return SubmitResult(ok=False, stderr="sbatch not found on this host")

    merged = os.environ.copy()
    merged.update(built.env_delta)

    try:
        proc = subprocess.run(
            built.argv, cwd=built.cwd, env=merged,
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        return SubmitResult(ok=False, stderr=str(exc))

    if proc.returncode != 0:
        return SubmitResult(ok=False, stdout=proc.stdout, stderr=proc.stderr)

    m = re.search(r"Submitted batch job (\d+)", proc.stdout)
    jobid = m.group(1) if m else ""
    outfile = ""
    if jobid and built.outfile_pattern:
        outfile = str(Path(built.cwd) / sbatch_header.resolve_outfile(
            built.outfile_pattern, jobid))
    return SubmitResult(ok=True, jobid=jobid, stdout=proc.stdout,
                        stderr=proc.stderr, outfile=outfile)
