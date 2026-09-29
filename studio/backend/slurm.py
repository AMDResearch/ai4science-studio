"""Slurm access via subprocess (list-form, shell=False).

Supports two modes:
  local  - sinfo/squeue/sbatch are on PATH (running on the cluster login node)
  remote - commands are proxied over SSH to ssh_host using the user's key agent

Degrades gracefully to available=False when neither is possible.
"""
from __future__ import annotations

import re
import shutil
import subprocess

_ALLOWED = re.compile(r"^[A-Za-z0-9._:+-]+$")
_TIMEOUT = 20

# SSH options: batch mode (no password prompt), strict host checking off for
# known internal hosts, 10s connect timeout.
_SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=10",
]


def _ssh_cmd(host: str, remote_cmd: list[str]) -> list[str]:
    """Wrap a command list for execution on a remote host via SSH.
    Passes the command as a single shell-quoted string so special chars
    (like | in sinfo format strings) are not interpreted by the remote shell."""
    import shlex
    return ["ssh"] + _SSH_OPTS + [host, " ".join(shlex.quote(a) for a in remote_cmd)]


def available(ssh_host: str | None = None) -> bool:
    if ssh_host:
        ok, _ = _run(["ssh"] + _SSH_OPTS + [ssh_host, "which", "sinfo"])
        return ok
    return shutil.which("sinfo") is not None


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if p.returncode != 0:
        return False, (p.stderr or p.stdout or f"exit {p.returncode}").strip()
    return True, p.stdout


def validate_token(value: str) -> bool:
    """Allowlist check for partition/account/qos/job-id values (anti-injection)."""
    return bool(value) and bool(_ALLOWED.match(value))


def partitions(ssh_host: str | None = None) -> dict:
    if not available(ssh_host):
        return {"available": False, "partitions": []}
    base = ["sinfo", "-h", "-o", "%P|%a|%l|%D|%G"]
    cmd = _ssh_cmd(ssh_host, base) if ssh_host else base
    ok, out = _run(cmd)
    if not ok:
        return {"available": True, "error": out, "partitions": []}
    parts = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 5:
            continue
        parts.append({
            "partition": f[0].strip(),
            "avail":     f[1].strip(),
            "timelimit": f[2].strip(),
            "nodes":     f[3].strip(),
            "gres":      f[4].strip(),
        })
    return {"available": True, "partitions": parts}


def nodes(ssh_host: str | None = None) -> dict:
    if not available(ssh_host):
        return {"available": False, "nodes": []}
    base = ["sinfo", "-N", "-h", "-o", "%N|%P|%t|%G|%m|%c"]
    cmd = _ssh_cmd(ssh_host, base) if ssh_host else base
    ok, out = _run(cmd)
    if not ok:
        return {"available": True, "error": out, "nodes": []}
    ns = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 6:
            continue
        ns.append({
            "node":      f[0].strip(),
            "partition": f[1].strip(),
            "state":     f[2].strip(),
            "gres":      f[3].strip(),
            "mem_mb":    f[4].strip(),
            "cpus":      f[5].strip(),
        })
    return {"available": True, "nodes": ns}


def queue(user: str | None = None, ssh_host: str | None = None) -> dict:
    if not available(ssh_host):
        return {"available": False, "jobs": []}
    base = ["squeue", "-h", "-o", "%i|%j|%T|%P|%M|%l|%D"]
    if user:
        base += ["-u", user]
    cmd = _ssh_cmd(ssh_host, base) if ssh_host else base
    ok, out = _run(cmd)
    if not ok:
        return {"available": True, "error": out, "jobs": []}
    jobs = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 7:
            continue
        jobs.append({
            "job_id":    f[0].strip(),
            "name":      f[1].strip(),
            "state":     f[2].strip(),
            "partition": f[3].strip(),
            "time":      f[4].strip(),
            "timelimit": f[5].strip(),
            "nodes":     f[6].strip(),
        })
    return {"available": True, "jobs": jobs}


def submit(script_path: str, ssh_host: str | None = None) -> tuple[bool, str]:
    """sbatch a script. Remote: copies script via ssh cat then sbatches it."""
    if ssh_host:
        # pipe script stdin to sbatch on the remote host
        try:
            with open(script_path) as fh:
                script_content = fh.read()
        except OSError as e:
            return False, str(e)
        try:
            p = subprocess.run(
                ["ssh"] + _SSH_OPTS + [ssh_host, "sbatch", "--parsable"],
                input=script_content, capture_output=True, text=True,
                timeout=_TIMEOUT, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        if p.returncode != 0:
            return False, (p.stderr or p.stdout or f"exit {p.returncode}").strip()
        return True, p.stdout.strip().split(";")[0]

    ok, out = _run(["sbatch", "--parsable", script_path])
    if not ok:
        return False, out
    return True, out.strip().split(";")[0]


def job_state(job_id: str, ssh_host: str | None = None) -> dict:
    """Query a job's state via sacct, falling back to squeue."""
    if not validate_token(job_id):
        return {"state": "UNKNOWN", "exit_code": None, "elapsed": None}
    if not available(ssh_host):
        return {"state": "UNKNOWN", "exit_code": None, "elapsed": None}

    base = ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode,Elapsed,NodeList"]
    cmd = _ssh_cmd(ssh_host, base) if ssh_host else base
    ok, out = _run(cmd)
    if not ok or not out.strip():
        # squeue fallback: also grab the node (%N) for live telemetry queries.
        base2 = ["squeue", "-h", "-j", job_id, "-o", "%T|%N"]
        cmd2 = _ssh_cmd(ssh_host, base2) if ssh_host else base2
        ok2, out2 = _run(cmd2)
        if ok2 and out2.strip():
            g = out2.strip().splitlines()[0].split("|")
            return {"state": g[0].strip(), "exit_code": None, "elapsed": None,
                    "node": _first_node(g[1] if len(g) > 1 else "")}
        return {"state": "UNKNOWN", "exit_code": None, "elapsed": None, "node": None}
    f = out.strip().splitlines()[0].split("|")
    return {
        "state":     f[0].strip() if len(f) > 0 else "UNKNOWN",
        "exit_code": f[1].strip() if len(f) > 1 else None,
        "elapsed":   f[2].strip() if len(f) > 2 else None,
        "node":      _first_node(f[3].strip() if len(f) > 3 else ""),
    }


def _first_node(nodelist: str) -> str | None:
    """First hostname from a SLURM NodeList (single-node runs → the node).

    Handles 'lux-mi355x-a2', 'lux-mi355x-a[2-5]', 'a2,a3'. Uses `scontrol show
    hostnames` when available for correctness on ranges; falls back to a simple parse.
    """
    nodelist = (nodelist or "").strip()
    if not nodelist or nodelist in ("None", "(null)", ""):
        return None
    if "[" in nodelist or "," in nodelist:
        ok, out = _run(["scontrol", "show", "hostnames", nodelist])
        if ok and out.strip():
            return out.strip().splitlines()[0]
    return nodelist.split(",")[0]


def cancel(job_id: str, ssh_host: str | None = None) -> tuple[bool, str]:
    if not validate_token(job_id):
        return False, "invalid job id"
    base = ["scancel", job_id]
    cmd = _ssh_cmd(ssh_host, base) if ssh_host else base
    return _run(cmd)
