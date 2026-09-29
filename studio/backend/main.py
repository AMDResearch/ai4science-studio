"""AMD AI4Science Studio — FastAPI backend."""
from __future__ import annotations

import getpass
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any

import jobs
import prompts
import provenance
import registry
import slurm
import video as video_mod

app = FastAPI(title="AMD AI4Science Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

provenance.init()
app.include_router(video_mod.router, prefix="/api")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"ok": True, "slurm": slurm.available(), "user": getpass.getuser()}


# ── Domain / Model Registry ───────────────────────────────────────────────────

@app.get("/api/domains")
def list_domains():
    return registry.list_domains()


@app.get("/api/domains/{domain}/models")
def list_models(domain: str):
    return registry.list_models(domain)


@app.get("/api/models/{slug}")
def get_model(slug: str):
    m = registry.get_model(slug)
    if not m:
        raise HTTPException(404, f"Model not found: {slug}")
    return m


@app.get("/api/models/{slug}/prompts")
def get_prompts(slug: str, domain: str = ""):
    return prompts.get_curated_prompts(slug, domain or None)


class ValidatePromptRequest(BaseModel):
    prompt: str
    domain: str

@app.post("/api/models/{slug}/validate-prompt")
def validate_prompt(slug: str, body: ValidatePromptRequest):
    return prompts.validate_prompt(slug, body.domain, body.prompt)


class AddModelRequest(BaseModel):
    slug: str
    name: str
    domain: str
    task: str
    hf_id: str = ""
    license: str = "Unknown"
    container_image: str | list[str] = ""
    vram_gb: float | None = None
    validated_hardware: list[str] = []
    tasks_available: list[str] = ["inference"]
    curated_prompts: list[dict] = []

def _is_demo_origin(request: Request) -> bool:
    o = (request.headers.get("origin") or request.headers.get("referer") or "")
    return "ai4science-studio-demo." in o or "-studio-demo" in o


@app.post("/api/models")
def add_model(body: AddModelRequest, request: Request):
    # Read-only demo site: no model registration (defense in depth).
    if _is_demo_origin(request):
        raise HTTPException(403, "Model registration is disabled on the read-only demo site.")
    ok = registry.add_model(body.model_dump())
    if not ok:
        raise HTTPException(500, "Failed to register model")
    return {"ok": True, "slug": body.slug}


# ── Jobs ──────────────────────────────────────────────────────────────────────

class LaunchRequest(BaseModel):
    slug: str
    domain: str
    task: str = "inference"
    mode: str = "demo"           # "demo" or "live"
    prompt: str
    params: dict[str, Any] = {}
    partition: str = "lux"

@app.post("/api/jobs")
def launch_job(body: LaunchRequest, request: Request):
    # Read-only demo site: reject LIVE launches whose Origin/Referer is the demo
    # hostname (defense in depth — the demo UI already grays out Live, but this blocks
    # a crafted request from the demo origin from ever submitting a SLURM job).
    if body.mode == "live" and _is_demo_origin(request):
        raise HTTPException(403, "Live mode is disabled on the read-only demo site.")
    # Validate prompt first
    v = prompts.validate_prompt(body.slug, body.domain, body.prompt)
    if not v["ok"]:
        raise HTTPException(422, v["message"])
    # Fail-closed: reject curated prompts that are grayed-out for this mode (the UI
    # disables them; this also blocks direct API calls from running dummy paths).
    if not prompts.prompt_allowed_in_mode(body.slug, body.prompt, body.mode):
        raise HTTPException(422, "This prompt is not available in the selected mode.")
    run_id = jobs.launch(
        slug=body.slug,
        domain=body.domain,
        task=body.task,
        mode=body.mode,
        prompt=body.prompt,
        params=body.params,
        partition=body.partition,
    )
    return {"run_id": run_id}


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{run_id}")
def get_job(run_id: str):
    job = jobs.get_job(run_id)
    if not job:
        raise HTTPException(404, f"Job not found: {run_id}")
    return job


@app.get("/api/jobs/{run_id}/log")
def get_log(run_id: str, lines: int = 200):
    return {"lines": jobs.tail_log(run_id, lines)}


@app.get("/api/jobs/{run_id}/files")
def get_files(run_id: str):
    return jobs.get_output_files(run_id)


@app.get("/api/jobs/{run_id}/files/{name}")
def get_file(run_id: str, name: str):
    from fastapi.responses import FileResponse
    p = jobs.get_output_file_path(run_id, name)
    if not p:
        raise HTTPException(404, f"File not found: {name}")
    return FileResponse(str(p), filename=p.name)


@app.get("/api/jobs/{run_id}/stream")
async def stream_job(run_id: str):
    return StreamingResponse(
        jobs.stream_log(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Telemetry (Omnistat) ────────────────────────────────────────────────────────

@app.get("/api/telemetry/catalog")
def telemetry_catalog():
    """The selectable Omnistat metric catalog for the Performance-tab dropdown."""
    import telemetry_catalog as cat
    return cat.as_json()


@app.get("/api/jobs/{run_id}/telemetry/live")
def telemetry_live(run_id: str, keys: str | None = None):
    """Live GPU telemetry from the RUNNING job's Omnistat VM on the compute node.

    Returns {status: pending|starting|live, telemetry?}. Omnistat only serves
    queryable data after its first ≥1-min push, so 'starting' means the exporter is
    up but no samples have landed yet (the UI shows a 'warming up' state).
    """
    key_list = [k for k in (keys.split(",") if keys else []) if k] or None
    return jobs.live_telemetry(run_id, key_list)


# ── Slurm info ────────────────────────────────────────────────────────────────

@app.get("/api/slurm/partitions")
def get_partitions():
    return slurm.partitions()


@app.get("/api/slurm/queue")
def get_queue():
    import getpass
    return slurm.queue(user=getpass.getuser())
