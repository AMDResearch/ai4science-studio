"""Read-only performance report agent.

Deterministic checks first; optional LLM ranking/explanation. Never submits jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s.llm import LLMError, chat_completion, provision_llm, stop_local_llm
from ai4s.records import artifact_entry, new_result, write_json
from ai4s.schema_util import validate


def run(context: dict[str, Any]) -> dict[str, Any]:
    run_id = str(context["run_id"])
    stage_dir = Path(context["stage_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)
    prior = context.get("prior_artifacts") or {}
    root = Path(context["root"])

    job_dir = _resolve_job_dir(prior, context.get("resolved_inputs") or {})
    if job_dir is None:
        return new_result(
            run_id=run_id,
            name=str(context["stage_id"]),
            status="error",
            error={
                "type": "MissingInput",
                "message": "no prior manifest/job dir; inference stage must succeed first",
            },
        )

    findings = _deterministic_findings(run_id, job_dir)
    synthesis = "deterministic_only"
    summary = _deterministic_summary(findings)
    llm_endpoint = None
    try:
        campaign_llm = (context.get("resolved_inputs") or {}).get("_campaign_llm") or {}
        # Only provision LLM if deterministic findings exist (or always try for ranking).
        llm_endpoint = provision_llm(campaign_llm, run_dir=stage_dir / "llm")
        ranked = _llm_synthesize(llm_endpoint, findings, job_dir)
        if ranked is not None:
            findings = ranked
            synthesis = "llm"
            summary = _deterministic_summary(findings)
    except LLMError as exc:
        synthesis = "failed"
        findings.append(
            {
                "id": "llm_synthesis_failed",
                "severity": "low",
                "title": "LLM synthesis unavailable",
                "detail": str(exc),
                "evidence": [{"path": str(job_dir / "manifest.json"), "quote": "deterministic findings retained"}],
                "source": "deterministic",
            }
        )
    finally:
        stop_local_llm(llm_endpoint)

    status = "partial" if synthesis == "failed" else "pass"
    payload = {
        "schema_version": "1",
        "run_id": run_id,
        "status": status,
        "synthesis": synthesis,
        "findings": findings,
        "summary": summary,
    }
    schema_errors = validate(payload, "findings.schema.json", root)
    if schema_errors:
        return new_result(
            run_id=run_id,
            name=str(context["stage_id"]),
            status="error",
            error={"type": "FindingsSchemaError", "message": "; ".join(schema_errors)},
        )

    findings_path = stage_dir / "findings.json"
    report_path = stage_dir / "perf_report.md"
    write_json(findings_path, payload)
    report_path.write_text(_render_markdown(payload, job_dir), encoding="utf-8")

    return new_result(
        run_id=run_id,
        name=str(context["stage_id"]),
        status=status,
        checks=[
            {
                "name": "deterministic_findings",
                "status": "pass",
                "detail": f"{len(findings)} finding(s)",
            },
            {
                "name": "llm_synthesis",
                "status": "pass" if synthesis == "llm" else ("fail" if synthesis == "failed" else "skip"),
                "detail": synthesis,
            },
        ],
        artifacts={
            "findings": artifact_entry(findings_path),
            "perf_report": artifact_entry(report_path),
        },
    )


def _resolve_job_dir(prior: dict[str, Any], inputs: dict[str, Any]) -> Path | None:
    for key in ("manifest", "orbit_inference.manifest", "job_dir"):
        entry = prior.get(key)
        if isinstance(entry, dict) and entry.get("path"):
            path = Path(entry["path"])
            if path.name == "manifest.json":
                return path.parent
            if path.is_dir():
                return path
    for key in ("ORBIT2_OUTPUT_DIR", "AI4S_OUTPUT_DIR", "job_dir"):
        if inputs.get(key):
            p = Path(str(inputs[key]))
            if (p / "manifest.json").is_file() or p.is_dir():
                return p
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _deterministic_findings(run_id: str, job_dir: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    manifest = _read_json(job_dir / "manifest.json")
    foms = _read_json(job_dir / "foms.json")

    if not manifest:
        findings.append(
            {
                "id": "missing_manifest",
                "severity": "high",
                "title": "Missing manifest.json",
                "detail": "Inference perf job did not write manifest.json",
                "evidence": [{"path": str(job_dir / "manifest.json")}],
                "source": "deterministic",
            }
        )
        return findings

    state = manifest.get("state") or manifest.get("status")
    if state and str(state).lower() not in {"ok", "succeeded", "success", "completed"}:
        findings.append(
            {
                "id": "job_state_not_ok",
                "severity": "high",
                "title": f"Job state is {state}",
                "detail": "Manifest reports a non-success terminal state",
                "evidence": [
                    {
                        "path": str(job_dir / "manifest.json"),
                        "quote": json.dumps({"state": state})[:200],
                    }
                ],
                "source": "deterministic",
            }
        )

    wall = foms.get("inference_wall_seconds") or manifest.get("inference_wall_seconds")
    if wall is None:
        findings.append(
            {
                "id": "missing_wall_time",
                "severity": "medium",
                "title": "Missing inference wall-clock FOM",
                "detail": "foms.json lacks inference_wall_seconds",
                "evidence": [{"path": str(job_dir / "foms.json")}],
                "source": "deterministic",
            }
        )
    else:
        findings.append(
            {
                "id": "inference_wall_time",
                "severity": "info",
                "title": "Inference wall time recorded",
                "detail": f"inference_wall_seconds={wall}",
                "evidence": [
                    {
                        "path": str(job_dir / "foms.json"),
                        "quote": f"inference_wall_seconds={wall}",
                    }
                ],
                "source": "deterministic",
            }
        )

    omni_db = job_dir / "omnistat-db"
    if not omni_db.exists():
        findings.append(
            {
                "id": "missing_omnistat_db",
                "severity": "medium",
                "title": "Omnistat database missing",
                "detail": "Telemetry directory omnistat-db was not found",
                "evidence": [{"path": str(omni_db)}],
                "source": "deterministic",
            }
        )
    else:
        findings.append(
            {
                "id": "omnistat_present",
                "severity": "info",
                "title": "Omnistat telemetry present",
                "detail": "omnistat-db directory exists",
                "evidence": [{"path": str(omni_db)}],
                "source": "deterministic",
            }
        )

    # Optional PromQL enrichments already in foms.json
    for key, title in (
        ("mfma_bf16_tflops_per_card_avg", "MFMA BF16 TFLOPS/card"),
        ("hbm_read_GBps_per_card_avg", "HBM read GB/s/card"),
    ):
        val = foms.get(key)
        if val is not None:
            findings.append(
                {
                    "id": f"metric_{key}",
                    "severity": "info",
                    "title": title,
                    "detail": f"{key}={val}",
                    "evidence": [
                        {
                            "path": str(job_dir / "foms.json"),
                            "quote": f"{key}={val}",
                        }
                    ],
                    "source": "deterministic",
                }
            )

    if not findings:
        findings.append(
            {
                "id": "no_issues",
                "severity": "info",
                "title": "No deterministic issues",
                "detail": f"run_id={run_id}",
                "evidence": [{"path": str(job_dir / "manifest.json")}],
                "source": "deterministic",
            }
        )
    return findings


def _deterministic_summary(findings: list[dict[str, Any]]) -> str:
    highs = [f for f in findings if f.get("severity") == "high"]
    if highs:
        return f"{len(highs)} high-severity finding(s); top: {highs[0]['title']}"
    return f"{len(findings)} finding(s); no high-severity issues"


def _llm_synthesize(
    endpoint: Any,
    findings: list[dict[str, Any]],
    job_dir: Path,
) -> list[dict[str, Any]] | None:
    """Ask LLM to rank/explain; keep only findings that cite existing evidence paths."""
    prompt = {
        "instructions": (
            "Rank and briefly explain these performance findings. "
            "Return JSON list with fields id,severity,title,detail,evidence,source. "
            "source must be 'llm'. Do not invent evidence paths; reuse provided ones. "
            "Do not recommend submitting jobs."
        ),
        "job_dir": str(job_dir),
        "findings": findings,
    }
    text = chat_completion(
        endpoint,
        messages=[
            {"role": "system", "content": "You are a performance analysis assistant."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
    )
    # Extract JSON array if wrapped in markdown fences.
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                cleaned = part
                break
    try:
        ranked = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(ranked, list):
        return None
    allowed_paths = {str(e.get("path")) for f in findings for e in f.get("evidence") or []}
    out: list[dict[str, Any]] = []
    for item in ranked:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") or []
        if not evidence:
            continue
        # Drop invented paths.
        evidence = [e for e in evidence if isinstance(e, dict) and str(e.get("path")) in allowed_paths]
        if not evidence:
            continue
        out.append(
            {
                "id": str(item.get("id") or f"llm_{len(out)}"),
                "severity": item.get("severity")
                if item.get("severity") in {"info", "low", "medium", "high"}
                else "info",
                "title": str(item.get("title") or "Untitled"),
                "detail": str(item.get("detail") or ""),
                "evidence": evidence,
                "source": "llm",
            }
        )
    return out or None


def _render_markdown(payload: dict[str, Any], job_dir: Path) -> str:
    lines = [
        "# Performance report",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- job_dir: `{job_dir}`",
        f"- synthesis: `{payload.get('synthesis')}`",
        f"- status: `{payload.get('status')}`",
        "",
        f"**Summary:** {payload.get('summary', '')}",
        "",
        "## Findings",
        "",
    ]
    for f in payload.get("findings") or []:
        lines.append(f"### [{f.get('severity')}] {f.get('title')} (`{f.get('id')}`)")
        lines.append("")
        if f.get("detail"):
            lines.append(str(f["detail"]))
            lines.append("")
        lines.append("Evidence:")
        for e in f.get("evidence") or []:
            quote = e.get("quote")
            if quote:
                lines.append(f"- `{e.get('path')}`: {quote}")
            else:
                lines.append(f"- `{e.get('path')}`")
        lines.append("")
    return "\n".join(lines) + "\n"
