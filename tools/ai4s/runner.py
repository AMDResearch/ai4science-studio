"""Allocation-aware command runner.

Uses ``srun`` when inside a SLURM allocation. Never nests ``sbatch``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class RunnerError(RuntimeError):
    pass


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    argv: list[str]


def in_slurm_allocation() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


def build_argv(
    command: Sequence[str] | str,
    *,
    use_srun: bool | None = None,
    srun_args: Sequence[str] | None = None,
) -> list[str]:
    if isinstance(command, str):
        base = shlex.split(command)
    else:
        base = list(command)
    if not base:
        raise RunnerError("empty command")
    # Refuse nested sbatch from the campaign runtime.
    if base[0] == "sbatch" or base[0].endswith("/sbatch"):
        raise RunnerError("campaign runner must not call sbatch; use srun or direct exec")
    if use_srun is None:
        use_srun = in_slurm_allocation()
    if not use_srun:
        return base
    prefix = ["srun", "--export=ALL"]
    if srun_args:
        prefix.extend(srun_args)
    return prefix + base


def run_command(
    command: Sequence[str] | str,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    use_srun: bool | None = None,
    srun_args: Sequence[str] | None = None,
    timeout: float | None = None,
    log_path: Path | None = None,
) -> CommandResult:
    argv = build_argv(command, use_srun=use_srun, srun_args=srun_args)
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=merged,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"command timed out after {timeout}s: {argv}") from exc
    result = CommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        argv=argv,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(shlex.quote(a) for a in argv)}\n"
            f"{result.stdout}{result.stderr}\n"
            f"EXIT={result.returncode}\n",
            encoding="utf-8",
        )
    return result
