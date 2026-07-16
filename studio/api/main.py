"""FastAPI app for AI4Science Studio.

Serves a JSON API over the reusable ``core`` package and (optionally) the built
React SPA from ``studio/web/dist``. Run with::

    uvicorn api.main:app --host 127.0.0.1 --port 8600

(with ``studio/`` on ``sys.path``; the launcher and the ``if __name__`` block
below both arrange this).

Every endpoint mirrors the behavior of the reference Streamlit pages in
``studio/pages/*.py`` — read those to understand the intended UX. In particular
the ``/run/*`` endpoints reproduce the Run page's blockers (disclaimer ack,
required fields, Apptainer-only guard, partition/account), the ORBIT-2
synthetic/real data-mode branch, AI4S_SHARED_DIR seeding, and the
"never emit empty optional env vars" rule.
"""
from __future__ import annotations

import hashlib
import socket
import sys
import time
from pathlib import Path
from typing import Any

# --- make `core` importable regardless of how uvicorn is launched -----------
_STUDIO_DIR = Path(__file__).resolve().parents[1]  # <repo>/studio
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from core import (  # noqa: E402
    catalog,
    cluster_config,
    form_builder,
    jobs,
    probes,
    scaffold,
    submission,
    suggestions,
    validation,
)
from core import sbatch_header as sbh  # noqa: E402

app = FastAPI(title="AI4Science Studio API", version="1.0.0")

# Permissive CORS so the Vite dev server (a different port) can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Serializers (dataclass -> plain dict) ------------------------------------
# ---------------------------------------------------------------------------

def _recipe_dict(r: catalog.Recipe) -> dict[str, Any]:
    return {
        "task": r.task,
        "description": r.description,
        "recipe_path": r.recipe_path,
        "script": r.script,
        "sbatch_script": r.sbatch_script,
        "runnable": r.runnable,
    }


def _model_summary(m: catalog.Model) -> dict[str, Any]:
    return {
        "slug": m.slug,
        "name": m.name,
        "domain": m.domain,
        "task": m.task,
        "license": m.license,
        "hf_id": m.hf_id,
        "tasks_available": m.tasks_available,
        "has_disclaimer": bool(m.disclaimer),
        "runnable_tasks": [r.task for r in m.recipes if r.runnable],
    }


def _model_detail(m: catalog.Model) -> dict[str, Any]:
    return {
        **_model_summary(m),
        "upstream_code": m.upstream_code,
        "paper": m.paper,
        "container_image": m.container_image,
        "vram_gb": m.vram_gb,
        "validated_hardware": m.validated_hardware,
        "disclaimer": m.disclaimer,
        "data_source": m.data_source,
        "weight_source": m.weight_source,
        "model_variants": m.model_variants,
        "recipes": [_recipe_dict(r) for r in m.recipes],
        "env_vars": m.env_vars,
        "schema_notes": validation.validate_model_yaml(m.raw),
    }


def _field_dict(f: form_builder.Field) -> dict[str, Any]:
    return {
        "name": f.name,
        "widget": f.widget,
        "default": f.default,
        "description": f.description,
        "required": f.required,
        "options": f.options,
        "is_path": f.is_path,
    }


def _discovery_dict(d: probes.Discovery) -> dict[str, Any]:
    return {
        "home": d.home,
        "gpu_arch": d.gpu_arch,
        "gpu_arch_label": probes.arch_label(d.gpu_arch) if d.gpu_arch else "",
        "gpu_count": d.gpu_count,
        "vram_gb": d.vram_gb,
        "rocm_version": d.rocm_version,
        "partitions": d.partitions,
        "accounts": d.accounts,
        "runtimes": d.runtimes,
        "scratch_dirs": d.scratch_dirs,
        "scratch_local": d.scratch_local,
        "net_ifaces": d.net_ifaces,
        "ib_hcas": d.ib_hcas,
        "internet": d.internet,
        "proxy": d.proxy,
        "sifs": d.sifs,
        "perf_tools": d.perf_tools,
    }


def _get_model_or_404(slug: str) -> catalog.Model:
    m = catalog.get_model(slug)
    if not m:
        raise HTTPException(status_code=404, detail=f"model not found: {slug}")
    return m


# ---------------------------------------------------------------------------
# Catalog / filters ---------------------------------------------------------
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models() -> dict[str, Any]:
    models = catalog.load_catalog()
    return {"models": [_model_summary(m) for m in models], "count": len(models)}


@app.get("/api/models/{slug}")
def get_model(slug: str) -> dict[str, Any]:
    return _model_detail(_get_model_or_404(slug))


@app.get("/api/filters")
def get_filters() -> dict[str, Any]:
    return {
        "domains": catalog.domains(),
        "tasks": catalog.all_tasks(),
        "licenses": catalog.all_licenses(),
    }


# ---------------------------------------------------------------------------
# Cluster config ------------------------------------------------------------
# ---------------------------------------------------------------------------

@app.get("/api/cluster")
def get_cluster() -> dict[str, Any]:
    active = cluster_config.active_config_path()
    return {
        "config": cluster_config.load_config(),
        "active_path": str(active) if active else None,
        "exists": active is not None,
        "repo_config_path": str(cluster_config.repo_config_path()),
        "user_config_path": str(cluster_config.user_config_path()),
        "runtime": cluster_config.runtime(),
        "hostname": socket.gethostname(),
        "can_submit": submission.can_submit(),
    }


@app.post("/api/cluster/discover")
def discover_cluster() -> dict[str, Any]:
    return _discovery_dict(probes.discover())


class ClusterSaveRequest(BaseModel):
    # Confirmed fields from the setup form (mirrors probes.to_config inputs).
    gpu_arch: str = ""
    gpu_count: int = 8
    vram_gb: int = 0
    partition: str = ""
    account: str = ""
    scratch: str = ""
    runtime: str = "apptainer"
    scratch_local: str = ""
    internet: bool = False
    proxy: str = ""
    net_ifaces: list[str] = Field(default_factory=list)
    ib_hcas: list[str] = Field(default_factory=list)
    perf_tools: str = ""
    location: str = cluster_config.REPO_CONFIG  # "repo" | "user"
    # Optionally post a fully-formed config dict instead of the fields above.
    config: dict[str, Any] | None = None


@app.post("/api/cluster")
def save_cluster(req: ClusterSaveRequest) -> dict[str, Any]:
    if req.config is not None:
        cfg = req.config
    else:
        d = probes.Discovery(
            gpu_arch=req.gpu_arch,
            gpu_count=req.gpu_count,
            vram_gb=req.vram_gb,
            scratch_local=req.scratch_local,
            internet=req.internet,
            proxy=req.proxy,
            net_ifaces=list(req.net_ifaces),
            ib_hcas=list(req.ib_hcas),
            perf_tools=req.perf_tools,
        )
        cfg = probes.to_config(
            d, partition=req.partition, account=req.account,
            scratch=req.scratch, runtime=req.runtime,
        )
    loc = req.location if req.location in (
        cluster_config.REPO_CONFIG, cluster_config.USER_CONFIG) else cluster_config.REPO_CONFIG
    path = cluster_config.save_config(cfg, location=loc)
    return {"ok": True, "path": str(path), "config": cfg,
            "schema_notes": validation.validate_cluster_config(cfg)}


# ---------------------------------------------------------------------------
# Run form (the key endpoint: gives React everything to avoid blank boxes) --
# ---------------------------------------------------------------------------

@app.get("/api/models/{slug}/recipes/{task}/form")
def get_run_form(slug: str, task: str) -> dict[str, Any]:
    m = _get_model_or_404(slug)
    recipe = m.recipe_for(task)
    if not recipe or not recipe.runnable:
        raise HTTPException(status_code=404,
                            detail=f"no runnable recipe '{task}' for {slug}")

    cfg = cluster_config.load_config()
    rt = cluster_config.runtime(cfg)
    root = suggestions.shared_root(cfg)
    header = sbh.parse(m.abs_path / recipe.sbatch_script)

    # ORBIT-2 data-mode branch (matches pages/4_run.py). The React form toggles
    # this client-side; we expose the flag + which vars to hide in each mode.
    orbit2 = None
    if slug == "ORBIT-2":
        orbit2 = {
            "synthetic_env": {"ORBIT2_USE_SYNTHETIC": "1"},
            "hidden_when_synthetic": ["ORBIT2_CONFIG", "ORBIT2_CHECKPOINT"],
            "hidden_when_real": ["ORBIT2_USE_SYNTHETIC"],
        }

    fields_out: list[dict[str, Any]] = []
    for f in form_builder.build_fields(m.env_vars):
        fd = _field_dict(f)
        if f.is_path:
            is_config = suggestions._kind(f.name) == "config"
            opts: list[str] = []
            if is_config and f.default:
                opts.append(f.default)
            if f.name == "AI4S_SHARED_DIR" and root and root not in opts:
                opts.append(root)
            for c in suggestions.candidates(m, f.name, cfg):
                if c not in opts:
                    opts.append(c)
            if f.default and f.default not in opts:
                opts.append(f.default)
            fd["suggestions"] = [
                {"value": o, "exists": (None if is_config
                                        else suggestions.path_exists(o))}
                for o in opts
            ]
            fd["is_config"] = is_config
        fields_out.append(fd)

    # AI4S_SHARED_DIR seed when the model doesn't declare it as an env var.
    shared_seed = None
    if root and not any(f["name"] == "AI4S_SHARED_DIR" for f in fields_out):
        shared_seed = root

    return {
        "slug": slug,
        "task": task,
        "recipe": _recipe_dict(recipe),
        "runtime": rt,
        "runtime_ok": (not rt) or rt in ("apptainer", "singularity"),
        "disclaimer": m.disclaimer,
        "scaling": {
            "nodes": header.nodes,
            "ntasks_per_node": header.ntasks_per_node,
            "gpus": header.gpus,
            "gpu_flag": header.gpu_flag,
            "time": header.time or "00:30:00",
        },
        "slurm_defaults": cluster_config.slurm_defaults(cfg),
        "shared_root": root,
        "shared_dir_seed": shared_seed,
        "fields": fields_out,
        "orbit2": orbit2,
        "can_submit": submission.can_submit(),
    }


# ---------------------------------------------------------------------------
# Run: preview + submit -----------------------------------------------------
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    slug: str
    task: str
    partition: str = ""
    account: str = ""
    qos: str = ""
    nodes: int = 1
    ntasks_per_node: int = 8
    gpus: int = 8
    time: str = ""
    env: dict[str, str] = Field(default_factory=dict)  # only vars to export
    disclaimer_ack: bool = False


def _build(req: RunRequest) -> tuple[catalog.Model, catalog.Recipe, submission.BuiltCommand]:
    m = _get_model_or_404(req.slug)
    recipe = m.recipe_for(req.task)
    if not recipe or not recipe.runnable:
        raise HTTPException(status_code=404,
                            detail=f"no runnable recipe '{req.task}' for {req.slug}")
    header = sbh.parse(m.abs_path / recipe.sbatch_script)
    # Never emit empty-string optional env vars (some scripts do int(VAR)).
    env = {k: v for k, v in req.env.items() if v != "" and v is not None}
    spec = submission.SubmitSpec(
        partition=req.partition, account=req.account, qos=req.qos,
        nodes=int(req.nodes), ntasks_per_node=int(req.ntasks_per_node),
        gpus=int(req.gpus), gpu_flag=header.gpu_flag,
        time=req.time, env=env,
    )
    built = submission.build_command(m, recipe, spec)
    return m, recipe, built


def _blockers(req: RunRequest, m: catalog.Model) -> list[str]:
    b: list[str] = []
    if m.disclaimer and not req.disclaimer_ack:
        b.append("acknowledge the disclaimer")
    if not submission.can_submit():
        b.append("sbatch not available on this host (dry-run only)")
    if not req.partition or not req.account:
        b.append("set SLURM partition and account")
    rt = cluster_config.runtime()
    if rt and rt not in ("apptainer", "singularity"):
        b.append(f"cluster runtime is '{rt}'; Studio runs Apptainer only")
    return b


@app.post("/api/run/preview")
def run_preview(req: RunRequest) -> dict[str, Any]:
    m, recipe, built = _build(req)
    outfile = ""
    if built.outfile_pattern:
        outfile = str(Path(built.cwd) / sbh.resolve_outfile(
            built.outfile_pattern, "<jobid>"))
    return {
        "preview": submission.render_preview(built),
        "cwd": built.cwd,
        "script_path": built.script_path,
        "outfile_pattern": built.outfile_pattern,
        "resolved_outfile": outfile,
        "argv": built.argv,
        "env_delta": built.env_delta,
        "can_submit": submission.can_submit(),
        "blockers": _blockers(req, m),
    }


@app.post("/api/run/submit")
def run_submit(req: RunRequest) -> dict[str, Any]:
    m, recipe, built = _build(req)
    blockers = _blockers(req, m)
    if blockers:
        raise HTTPException(status_code=400,
                            detail={"error": "cannot submit", "blockers": blockers})

    result = submission.submit(built)
    if not result.ok:
        return JSONResponse(status_code=502, content={
            "ok": False, "stderr": result.stderr, "stdout": result.stdout})

    rt = cluster_config.runtime()
    rec: dict[str, Any] = {
        "jobid": result.jobid,
        "model": req.slug,
        "recipe": req.task,
        "runtime": rt or "apptainer",
        "submit_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": built.cwd,
        "script": built.script_path,
        "outfile": result.outfile,
        "params": built.env_delta,
        "overrides": {
            "nodes": int(req.nodes), "ntasks_per_node": int(req.ntasks_per_node),
            "gpus": int(req.gpus), "partition": req.partition,
            "account": req.account,
        },
        "last_state": "PENDING",
    }
    if m.disclaimer:
        rec["disclaimer_ack"] = {
            "hash": hashlib.sha256(m.disclaimer.encode()).hexdigest()[:16],
            "ts": rec["submit_time"],
        }
    jobs.add_job(rec)
    return {"ok": True, "jobid": result.jobid, "outfile": result.outfile,
            "stdout": result.stdout}


# ---------------------------------------------------------------------------
# Jobs ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    records = jobs.list_jobs()
    states = jobs.poll_states()
    out: list[dict[str, Any]] = []
    for rec in records:
        jid = rec.get("jobid", "")
        state = states.get(jid, rec.get("last_state", "UNKNOWN"))
        outfile = rec.get("outfile", "")
        metrics = jobs.extract_metrics(outfile) if outfile else {}
        out.append({
            **rec,
            "state": state,
            "exit_code": rec.get("exit_code", ""),
            "metrics": metrics,
        })
    return {"jobs": out}


@app.get("/api/jobs/{jobid}/log")
def job_log(jobid: str, max_bytes: int = 60_000) -> dict[str, Any]:
    for rec in jobs.list_jobs():
        if rec.get("jobid") == jobid:
            outfile = rec.get("outfile", "")
            return {"jobid": jobid, "outfile": outfile,
                    "log": jobs.tail_log(outfile, max_bytes) if outfile else ""}
    raise HTTPException(status_code=404, detail=f"job not found: {jobid}")


@app.post("/api/jobs/{jobid}/cancel")
def job_cancel(jobid: str) -> dict[str, Any]:
    ok = jobs.cancel(jobid)
    return {"ok": ok, "jobid": jobid}


@app.delete("/api/jobs/{jobid}")
def job_remove(jobid: str) -> dict[str, Any]:
    jobs.remove_job(jobid)
    return {"ok": True, "jobid": jobid}


# ---------------------------------------------------------------------------
# Add model (scaffold) ------------------------------------------------------
# ---------------------------------------------------------------------------

class NewModelRequest(BaseModel):
    slug: str
    domain: str
    hf_id: str = ""
    license: str = ""
    task: str = ""
    upstream_code: str = ""
    paper: str = ""
    container_image: str = ""


@app.get("/api/scaffold/domains")
def scaffold_domains() -> dict[str, Any]:
    return {"domains": scaffold.DOMAINS}


@app.post("/api/models")
def create_model(req: NewModelRequest) -> dict[str, Any]:
    nm = scaffold.NewModel(
        slug=req.slug, domain=req.domain, hf_id=req.hf_id,
        license=req.license, task=req.task, upstream_code=req.upstream_code,
        paper=req.paper, container_image=req.container_image,
    )
    errs = scaffold.validate(nm)
    if errs:
        raise HTTPException(status_code=400,
                            detail={"error": "validation failed", "errors": errs})
    try:
        dest = scaffold.create(nm)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "path": str(dest)}


# ---------------------------------------------------------------------------
# Static SPA (served only if a build exists) --------------------------------
# ---------------------------------------------------------------------------

_DIST = _STUDIO_DIR / "web" / "dist"


@app.get("/api")
def api_index() -> dict[str, Any]:
    return {
        "name": "AI4Science Studio API",
        "version": app.version,
        "frontend_built": _DIST.is_dir(),
        "endpoints": [
            "GET /api/models", "GET /api/models/{slug}", "GET /api/filters",
            "GET /api/cluster", "POST /api/cluster/discover", "POST /api/cluster",
            "GET /api/models/{slug}/recipes/{task}/form",
            "POST /api/run/preview", "POST /api/run/submit",
            "GET /api/jobs", "GET /api/jobs/{jobid}/log",
            "POST /api/jobs/{jobid}/cancel", "DELETE /api/jobs/{jobid}",
            "GET /api/scaffold/domains", "POST /api/models",
        ],
    }


if _DIST.is_dir():
    # html=True serves index.html for unknown paths -> SPA client routing works.
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
else:
    @app.get("/")
    def root_index() -> dict[str, Any]:
        return {
            "name": "AI4Science Studio API",
            "frontend_built": False,
            "note": "React frontend not built. Build studio/web (npm install && "
                    "npm run build) to produce studio/web/dist, or use the Vite "
                    "dev server. See /api for the endpoint list.",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8600)
