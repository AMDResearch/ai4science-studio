"""LLM client with explicit external/local/auto routing and two wire providers.

Providers: ``openai`` (``/chat/completions``) and ``anthropic`` (Messages API, e.g. the AMD
Gateway's ``/Anthropic/v1/messages`` for Claude). LiteLLM is the planned next step for
broader provider coverage; until then the two providers are plain stdlib HTTP.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


PROVIDERS = {"openai", "anthropic"}
ANTHROPIC_VERSION = "2023-06-01"


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
    provider: str = "openai"  # openai | anthropic
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
    # AI4S_LLM_PROVIDER / AI4S_LLM_MODEL override the campaign file (per-deployment choice).
    provider = str(os.environ.get("AI4S_LLM_PROVIDER") or cfg.get("provider") or "openai").lower()
    if provider not in PROVIDERS:
        raise LLMError(f"invalid llm.provider: {provider}")
    return {
        "mode": mode,
        "provider": provider,
        "endpoint": cfg.get("endpoint") or os.environ.get("AI4S_LLM_ENDPOINT", ""),
        "api_key_env": cfg.get("api_key_env") or "OPENAI_API_KEY",
        "auth_style": auth_style,
        "model": os.environ.get("AI4S_LLM_MODEL") or cfg.get("model") or "gpt-oss-20b",
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


def _external_endpoint(cfg: dict[str, Any], base_url: str) -> LLMEndpoint:
    ep = LLMEndpoint(
        mode="external",
        base_url=base_url,
        model=str(cfg["model"]),
        api_key_env=str(cfg["api_key_env"]),
        auth_style=str(cfg.get("auth_style") or "auto"),
        provider=str(cfg.get("provider") or "openai"),
    )
    wait_external_ready(ep, float(os.environ.get("AI4S_LLM_EXTERNAL_READY_TIMEOUT", "0")))
    return ep


def wait_external_ready(endpoint: LLMEndpoint, timeout: float) -> None:
    """Bounded gate on the relay's ``/healthz`` (origin root); raises LLMError so callers degrade.

    Off unless ``AI4S_LLM_EXTERNAL_READY_TIMEOUT`` > 0; only meaningful for the AI4S relay.
    """
    if timeout <= 0:
        return
    parts = urllib.parse.urlsplit(endpoint.base_url)
    url = f"{parts.scheme}://{parts.netloc}/healthz"
    req = urllib.request.Request(url, headers=auth_headers(endpoint), method="GET")
    deadline = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if time.time() >= deadline:
            raise LLMError(f"LLM endpoint not ready after {timeout:.0f}s: {url}")
        time.sleep(2)


def _auto_candidates(cfg: dict[str, Any]) -> list[str]:
    """Ordered external endpoints to probe when mode=auto and no explicit endpoint is set.

    Priority:
      tunnel (via http://127.0.0.1:<port>) where port is
        $AI4S_RELAY_COMPUTE_PORT | $COMPUTE_PORT | $AI4S_LLM_TUNNEL_PORT | 18000
        — hinted by $AI4S_RELAY_CLIENT_ENV / $AI4S_LLM_TUNNEL_PORT but also probed
        opportunistically if /healthz responds quickly;
      on-prem fallback via $AI4S_LLM_ON_PREM_ENDPOINT (site-configurable, e.g. Muse Spark URL).
    """
    candidates: list[str] = []
    # Relay/tunnel port: explicit endpoint already handled by caller; here we discover
    # compute-local relay forwarded via ssh -R to login node (port 18000 by default).
    # Env precedence: AI4S_RELAY_COMPUTE_PORT (new) > COMPUTE_PORT (legacy) > AI4S_LLM_TUNNEL_PORT > 18000.
    port_raw = (
        os.environ.get("AI4S_RELAY_COMPUTE_PORT")
        or os.environ.get("COMPUTE_PORT")
        or os.environ.get("AI4S_LLM_TUNNEL_PORT")
        or "18000"
    )
    try:
        port = int(str(port_raw).strip())
        if 1 <= port <= 65535:
            provider = str(cfg.get("provider") or "openai").lower()
            prefix = "Anthropic" if provider == "anthropic" else "OpenAI"
            # Relay mounts gateway under /<Prefix> (e.g. /OpenAI or /Anthropic); generic vLLM under /v1.
            # Try provider-specific first, then generic, then bare host (for healthz-only gateways).
            candidates.extend(
                [
                    f"http://127.0.0.1:{port}/{prefix}",
                    f"http://127.0.0.1:{port}/v1",
                    f"http://127.0.0.1:{port}",
                ]
            )
    except (ValueError, TypeError):
        pass
    on_prem = os.environ.get("AI4S_LLM_ON_PREM_ENDPOINT", "").strip().rstrip("/")
    if on_prem:
        candidates.append(on_prem)
    return candidates


def provision_llm(
    campaign_llm: dict[str, Any] | None,
    *,
    run_dir: Path,
    start_local: Callable[[dict[str, Any], Path], LLMEndpoint] | None = None,
) -> LLMEndpoint:
    """Resolve endpoint. Never silently falls back to hosted models.

    Modes:
      external: requires an explicit endpoint (cfg["endpoint"] or $AI4S_LLM_ENDPOINT),
        else error; still gates on wait_external_ready if $AI4S_LLM_EXTERNAL_READY_TIMEOUT>0.
      local: always launch local vLLM.
      auto: health-probe fallback without adding a new mode:
        1. explicit endpoint (cfg["endpoint"] or $AI4S_LLM_ENDPOINT) if set -> use it
        2. tunnel probe: http://127.0.0.1:<port> where port is
           $AI4S_RELAY_COMPUTE_PORT | $COMPUTE_PORT | $AI4S_LLM_TUNNEL_PORT | 18000
           — hinted by $AI4S_RELAY_CLIENT_ENV / $AI4S_LLM_TUNNEL_PORT but also probed
           opportunistically if /healthz responds quickly (short timeout via _health_ok)
        3. on-prem fallback: $AI4S_LLM_ON_PREM_ENDPOINT if set (site-configurable,
           e.g. Muse Spark URL)
        4. otherwise fallback to local vLLM; do not fail if probes miss.
    """
    cfg = resolve_llm_config(campaign_llm)
    mode = cfg["mode"]
    endpoint = str(cfg["endpoint"] or "").rstrip("/")
    has_external = bool(endpoint)

    if mode == "external":
        if not has_external:
            raise LLMError("llm.mode=external requires endpoint or AI4S_LLM_ENDPOINT")
        return _external_endpoint(cfg, endpoint)

    if mode == "local":
        starter = start_local or start_local_vllm
        return starter(cfg, run_dir)

    # auto
    if has_external:
        return _external_endpoint(cfg, endpoint)
    # No explicit endpoint: probe tunnel then on-prem in order; first healthy wins.
    for cand in _auto_candidates(cfg):
        # _auto_candidates already orders tunnel variants then on-prem
        ep = LLMEndpoint(
            mode="external",
            base_url=cand,
            model=str(cfg["model"]),
            api_key_env=str(cfg["api_key_env"]),
            auth_style=str(cfg.get("auth_style") or "auto"),
            provider=str(cfg.get("provider") or "openai"),
        )
        # Use _health_ok with short timeout (2s) — if healthz/models responds, cand is reachable.
        # If not (no relay, no on-prem), try next; do not fail, fall through to local.
        if _health_ok(cand, ep):
            return _external_endpoint(cfg, cand)
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


def _build_request(
    endpoint: LLMEndpoint,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Return (url, payload, extra headers) for the endpoint's provider."""
    base = endpoint.base_url.rstrip("/")
    if endpoint.provider == "anthropic":
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        url, extra = base + "/v1/messages", {"anthropic-version": ANTHROPIC_VERSION}
    else:
        payload = {
            "model": endpoint.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url, extra = base + "/chat/completions", {}
    if stream:
        payload["stream"] = True
    return url, payload, extra


def _open(endpoint: LLMEndpoint, url: str, payload: dict[str, Any], extra: dict[str, str]):
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
    headers = {"Content-Type": "application/json", **extra, **auth_headers(endpoint)}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        return urllib.request.urlopen(req, timeout=120)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc


def _text_of(provider: str, body: dict[str, Any]) -> str:
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in body["content"] if b.get("type") == "text")
    return str(body["choices"][0]["message"]["content"])


def chat_completion(
    endpoint: LLMEndpoint,
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    url, payload, extra = _build_request(endpoint, messages, temperature, max_tokens, False)
    try:
        with _open(endpoint, url, payload, extra) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    try:
        return _text_of(endpoint.provider, body)
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected LLM response shape: {body!r}") from exc


def _sse_events(lines: Iterable[bytes]) -> Iterator[tuple[str, str]]:
    """Yield (event, data) from an SSE byte-line stream; ``event`` is '' when absent."""
    event, data = "", []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, "\n".join(data)


def _delta_text(provider: str, event: str, data: str) -> str | None:
    """Text delta carried by one SSE event, or None; raises LLMError on a stream error."""
    if data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as exc:
        raise LLMError(f"malformed stream event: {data[:200]!r}") from exc
    if event == "error" or obj.get("type") == "error":
        raise LLMError(f"LLM stream error: {obj.get('error', obj)}")
    if provider == "anthropic":
        if obj.get("type") == "content_block_delta" and obj["delta"].get("type") == "text_delta":
            return str(obj["delta"]["text"])
        return None
    choices = obj.get("choices") or []
    return (choices[0].get("delta") or {}).get("content") or None if choices else None


def stream_chat(
    endpoint: LLMEndpoint,
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Yield text deltas as they arrive; works for both providers."""
    url, payload, extra = _build_request(endpoint, messages, temperature, max_tokens, True)
    with _open(endpoint, url, payload, extra) as resp:
        try:
            for event, data in _sse_events(resp):
                text = _delta_text(endpoint.provider, event, data)
                if text:
                    yield text
        except (TimeoutError, OSError) as exc:
            raise LLMError(f"LLM stream interrupted: {exc}") from exc
