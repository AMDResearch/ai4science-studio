"""Cluster auto-discovery probes.

Python port of the ~14 discovery commands from .claude/commands/init-cluster.md.
Every probe is best-effort: on any error it returns an empty/blank value rather
than raising, so the setup page can render partial results.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


@dataclass
class Discovery:
    home: str = ""
    gpu_arch: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    rocm_version: str = ""
    partitions: list[str] = field(default_factory=list)
    accounts: list[str] = field(default_factory=list)
    runtimes: list[str] = field(default_factory=list)
    scratch_dirs: list[str] = field(default_factory=list)
    scratch_local: str = ""
    net_ifaces: list[str] = field(default_factory=list)
    ib_hcas: list[str] = field(default_factory=list)
    internet: bool = False
    proxy: str = ""
    sifs: list[str] = field(default_factory=list)
    perf_tools: str = ""


_ARCH_NAMES = {
    "gfx942": "MI300X",
    "gfx90a": "MI250X / MI210",
    "gfx950": "MI355X",
    "gfx908": "MI100",
}


def arch_label(gfx: str) -> str:
    name = _ARCH_NAMES.get(gfx)
    return f"{gfx} ({name})" if name else gfx


def discover() -> Discovery:
    d = Discovery()

    d.home = os.environ.get("HOME", "")

    # GPU architecture
    ri = _run(["rocminfo"], timeout=15)
    archs = sorted(set(re.findall(r"gfx\d+", ri)))
    if archs:
        d.gpu_arch = archs[0]

    # GPU count from render nodes
    d.gpu_count = len(glob.glob("/dev/dri/renderD*"))

    # VRAM
    vram = _run(["rocm-smi", "--showmeminfo", "vram"], timeout=15)
    mb = re.search(r"Total.*?:\s*(\d+)", vram)
    if mb:
        d.vram_gb = round(int(mb.group(1)) / (1024**3)) if int(mb.group(1)) > 10**6 else round(int(mb.group(1)) / 1024)

    # ROCm version
    if os.path.exists("/opt/rocm/.info/version"):
        try:
            d.rocm_version = open("/opt/rocm/.info/version").read().strip()
        except Exception:  # noqa: BLE001
            pass

    # SLURM partitions (GPU)
    sinfo = _run(["sinfo", "-h", "-o", "%P %G"], timeout=10)
    for line in sinfo.splitlines():
        parts = line.split()
        if parts and ("gpu" in line.lower()):
            d.partitions.append(parts[0].rstrip("*"))
    d.partitions = sorted(set(d.partitions))

    # SLURM accounts
    user = os.environ.get("USER", "")
    if user:
        assoc = _run(
            ["sacctmgr", "show", "associations", f"where", f"user={user}",
             "format=account%30", "-n"],
            timeout=10,
        )
        d.accounts = sorted({a.strip() for a in assoc.split() if a.strip()})

    # Container runtimes
    if shutil.which("apptainer"):
        d.runtimes.append("apptainer")
    if shutil.which("singularity"):
        d.runtimes.append("singularity")
    if shutil.which("docker"):
        d.runtimes.append("docker")

    # Scratch dirs
    for base in (f"/scratch/{user}", "/scratch", "/lustre", "/gpfs",
                 f"/shared/{user}", "/shared", f"/tmp/{user}"):
        if base and os.path.isdir(base):
            d.scratch_dirs.append(base)

    # Node-local fast storage
    for base in ("/scratch", "/local", "/nvme", "/tmp"):
        if os.path.isdir(base) and os.access(base, os.W_OK):
            d.scratch_local = base
            break

    # Network interfaces
    ip = _run(["ip", "-o", "link", "show", "up"], timeout=5)
    for line in ip.splitlines():
        m = re.match(r"\d+:\s*([^:@]+)", line)
        if m:
            name = m.group(1).strip()
            if name != "lo":
                d.net_ifaces.append(name)

    # IB / RoCE HCAs
    ibstat = _run(["ibstat"], timeout=10)
    d.ib_hcas = re.findall(r"CA '([^']+)'", ibstat)

    # Internet
    code = _run(
        ["curl", "-s", "--connect-timeout", "5", "-o", "/dev/null",
         "-w", "%{http_code}", "https://huggingface.co"],
        timeout=10,
    )
    d.internet = code.startswith("2") or code.startswith("3")

    # Proxy
    d.proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""

    # SIF files
    search_roots = [d.home] + d.scratch_dirs[:3] + ["/opt"]
    for root in search_roots:
        if root and os.path.isdir(root):
            found = _run(
                ["find", root, "-maxdepth", "4", "-name", "*.sif"],
                timeout=20,
            )
            d.sifs.extend([s for s in found.splitlines() if s])
    d.sifs = sorted(set(d.sifs))[:20]

    # Perf tooling
    for base in glob.glob("/shared/*/tools") + ["/shared/perf-tools",
                                                 f"{d.home}/perf-tools"]:
        if os.path.isdir(os.path.join(base, "perf-inspect")) or os.path.isdir(
            os.path.join(base, "omnistat-src")
        ):
            d.perf_tools = base
            break

    return d


def to_config(d: Discovery, partition: str, account: str, scratch: str,
              runtime: str) -> dict:
    """Assemble a .cluster-config.yaml dict from a Discovery + user choices."""
    return {
        "slurm": {"partition": partition, "account": account, "qos": ""},
        "paths": {
            "home": "",  # blank => use $HOME
            "scratch": scratch,
            "scratch_local": d.scratch_local,
            "projects": f"{scratch}/models" if scratch else "",
            "sif_cache": f"{scratch}/images" if scratch else "",
        },
        "containers": {"runtime": runtime, "gpu_method": ""},
        "gpu": {
            "architecture": d.gpu_arch,
            "count_per_node": d.gpu_count or 8,
            "vram_gb": d.vram_gb,
        },
        "network": {
            "internet_access": d.internet,
            "proxy": d.proxy,
            "mgmt_iface": d.net_ifaces[0] if d.net_ifaces else "",
            "ib_hca": ",".join(d.ib_hcas),
            "rccl_anp_plugin": "",
            "libionic_path": "",
        },
        "perf_tools": {"dir": d.perf_tools},
    }
