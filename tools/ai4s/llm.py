"""OpenAI-compatible LLM client with explicit external/local/auto routing."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class LLMError(RuntimeError):
    pass


@dataclass
class LLMEndpoint:
    mode: str  # external | local
    base_url: str
    model: str
    api_key_env: str | None = None
    # bearer | apim | auto — apim sends Ocp-Apim-Subscription-Key (AMD llm-api.amd.com)
    auth_style: str = "auto"
    local_pid: int | None = None
    local_log: Path | None = None


def resolve_llm_config(campaign_llm: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(campaign_llm or {})
    mode = str(cfg.get("mode") or "auto")
    if mode not in {"external", "local", "auto"}:
        raise LLMError(f"invalid llm.mode: {mode}")
    auth_style = str(
        cfg.get("auth_style") or os.environ.get("AI4S_LLM_AUTH", "auto")
    ).lower()
    if auth_style not in {"auto", "bearer", "apim"}:
        raise LLMError(f"invalid llm.auth_style: {auth_style}")
    return {
        "mode": mode,
        "endpoint": cfg.get("endpoint") or os.environ.get("AI4S_LLM_ENDPOINT", ""),
        "api_key_env": cfg.get("api_key_env") or "OPENAI_API_KEY",
        "auth_style": auth_style,
        "model": cfg.get("model") or os.environ.get("AI4S_LLM_MODEL", "gpt-oss-20b"),
        "local": cfg.get("local") or {},
    }


def _resolved_auth_style(endpoint: LLMEndpoint) -> str:
    style = (endpoint.auth_style or "auto").lower()
    if style in {"bearer", "apim"}:
        return style
    # auto: AMD Model Access Gateway / APIM uses subscription-key header, not Bearer.
    if "llm-api.amd.com" in (endpoint.base_url or "").lower():
        return "apim"
    return "bearer"


def auth_headers(endpoint: LLMEndpoint) -> dict[str, str]:
    """Build auth headers. Never logs or returns the key value to callers for printing."""
    if not endpoint.api_key_env:
        return {}
    key = os.environ.get(endpoint.api_key_env, "").strip()
    if not key:
        return {}
    style = _resolved_auth_style(endpoint)
    if style == "apim":
        return {"Ocp-Apim-Subscription-Key": key}
    return {"Authorization": f"Bearer {key}"}


def provision_llm(
    campaign_llm: dict[str, Any] | None,
    *,
    run_dir: Path,
    start_local: Callable[[dict[str, Any], Path], LLMEndpoint] | None = None,
) -> LLMEndpoint:
    """Resolve endpoint. Never silently falls back to hosted models."""
    cfg = resolve_llm_config(campaign_llm)
    mode = cfg["mode"]
    endpoint = str(cfg["endpoint"] or "").rstrip("/")
    has_external = bool(endpoint)

    if mode == "external":
        if not has_external:
            raise LLMError("llm.mode=external requires endpoint or AI4S_LLM_ENDPOINT")
        return LLMEndpoint(
            mode="external",
            base_url=endpoint,
            model=str(cfg["model"]),
            api_key_env=str(cfg["api_key_env"]),
            auth_style=str(cfg.get("auth_style") or "auto"),
        )

    if mode == "local":
        starter = start_local or start_local_vllm
        return starter(cfg, run_dir)

    # auto
    if has_external:
        return LLMEndpoint(
            mode="external",
            base_url=endpoint,
            model=str(cfg["model"]),
            api_key_env=str(cfg["api_key_env"]),
            auth_style=str(cfg.get("auth_style") or "auto"),
        )
    starter = start_local or start_local_vllm
    return starter(cfg, run_dir)


def start_local_vllm(cfg: dict[str, Any], run_dir: Path) -> LLMEndpoint:
    """Start a same-allocation local OpenAI-compatible server (vLLM via apptainer/python).

    This is intentionally conservative: if no local image/command is configured,
    raise a clear error instead of inventing a hosted fallback.
    """
    local = cfg.get("local") or {}
    image = local.get("image") or os.environ.get("AI4S_VLLM_IMAGE", "")
    model = local.get("model") or cfg.get("model") or os.environ.get("AI4S_LLM_MODEL", "")
    port = int(local.get("port") or os.environ.get("AI4S_VLLM_PORT", "8000"))
    if not image or not model:
        raise LLMError(
            "local LLM requested but image/model not configured "
            "(set campaign llm.local.image/model or AI4S_VLLM_IMAGE / AI4S_LLM_MODEL)"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "local_llm.log"
    base_url = f"http://127.0.0.1:{port}/v1"
    # Prefer apptainer when a SIF/image path is given; otherwise try python -m vllm.
    if str(image).endswith(".sif") or Path(str(image)).exists():
        cmd = [
            "apptainer",
            "exec",
            "--rocm",
            str(image),
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ]
    else:
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ]

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    deadline = time.time() + int(os.environ.get("AI4S_LLM_READY_TIMEOUT", "300"))
    while time.time() < deadline:
        if proc.poll() is not None:
            raise LLMError(f"local LLM exited early; see {log_path}")
        if _health_ok(base_url):
            return LLMEndpoint(
                mode="local",
                base_url=base_url,
                model=str(model),
                api_key_env=None,
                local_pid=proc.pid,
                local_log=log_path,
            )
        time.sleep(2)
    proc.terminate()
    raise LLMError(f"local LLM did not become healthy; see {log_path}")


def stop_local_llm(endpoint: LLMEndpoint | None) -> None:
    if endpoint is None or endpoint.local_pid is None:
        return
    try:
        os.kill(endpoint.local_pid, 15)
    except OSError:
        pass


def _health_ok(base_url: str, endpoint: LLMEndpoint | None = None) -> bool:
    url = base_url.rstrip("/") + "/models"
    headers = auth_headers(endpoint) if endpoint is not None else {}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def chat_completion(
    endpoint: LLMEndpoint,
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": endpoint.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(endpoint))
    if endpoint.api_key_env and endpoint.api_key_env not in os.environ:
        raise LLMError(
            f"LLM auth env {endpoint.api_key_env} is unset "
            f"(auth_style={_resolved_auth_style(endpoint)})"
        )
    if endpoint.api_key_env and not os.environ.get(endpoint.api_key_env, "").strip():
        raise LLMError(
            f"LLM auth env {endpoint.api_key_env} is empty "
            f"(auth_style={_resolved_auth_style(endpoint)})"
        )
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected LLM response shape: {body!r}") from exc
