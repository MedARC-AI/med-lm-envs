"""Health checks for local vLLM-backed bench runs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

HealthCheckMode = Literal["auto", "on", "off"]
OPENAI_CHAT_CLIENT_TYPES = {"openai_chat_completions", "openai_chat_completions_token", "nemorl_chat_completions"}
LOCAL_VLLM_PROVIDERS = {"local", "vllm"}
DEFAULT_VLLM_HEALTH_CHECK_INTERVAL_SECONDS = 600.0
DEFAULT_VLLM_HEALTH_CHECK_TIMEOUT_SECONDS = 120.0
DEFAULT_VLLM_HEALTH_CHECK_FAILURES = 2


@dataclass(frozen=True)
class VllmHealthCheckConfig:
    api_base_url: str
    model: str
    api_key_var: str | None = None
    extra_headers: dict[str, str] | None = None
    interval_seconds: float = DEFAULT_VLLM_HEALTH_CHECK_INTERVAL_SECONDS
    timeout_seconds: float = DEFAULT_VLLM_HEALTH_CHECK_TIMEOUT_SECONDS
    failure_threshold: int = DEFAULT_VLLM_HEALTH_CHECK_FAILURES


class VllmHealthCheckError(RuntimeError):
    """Raised when a local vLLM endpoint fails repeated health probes."""


def resolve_vllm_health_check_config(
    eval_config: Any,
    *,
    provider: str | None,
    mode: HealthCheckMode,
    interval_seconds: float,
    timeout_seconds: float,
    failure_threshold: int,
) -> VllmHealthCheckConfig | None:
    """Return a health-check config when the resolved bench target is local vLLM-like."""

    if mode == "off":
        return None

    client_config = getattr(eval_config, "client_config", None)
    client_type = str(getattr(client_config, "client_type", "") or "")
    api_base_url = str(getattr(client_config, "api_base_url", "") or "")
    model = str(getattr(eval_config, "model", "") or "")

    if not api_base_url or not model or client_type not in OPENAI_CHAT_CLIENT_TYPES:
        return None

    is_local_vllm = (provider or "").strip().lower() in LOCAL_VLLM_PROVIDERS or _is_loopback_url(api_base_url)
    if mode == "auto" and not is_local_vllm:
        return None

    extra_headers = {
        str(key): str(value)
        for key, value in dict(getattr(client_config, "extra_headers", {}) or {}).items()
        if value is not None
    }
    return VllmHealthCheckConfig(
        api_base_url=api_base_url,
        model=model,
        api_key_var=str(getattr(client_config, "api_key_var", "") or "") or None,
        extra_headers=extra_headers,
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
        failure_threshold=failure_threshold,
    )


async def run_with_vllm_health_check(
    eval_coro_factory: Callable[[], Awaitable[Any]],
    health_config: VllmHealthCheckConfig | None,
) -> Any:
    """Run an eval coroutine while monitoring its local vLLM endpoint."""

    if health_config is None:
        return await eval_coro_factory()

    eval_task = asyncio.create_task(eval_coro_factory())
    monitor_task = asyncio.create_task(_monitor_vllm_endpoint(health_config))
    done, pending = await asyncio.wait({eval_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

    try:
        for task in done:
            exc = task.exception()
            if exc is not None:
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise exc

        if eval_task in done:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task
            return eval_task.result()

        return await eval_task
    finally:
        for task in (eval_task, monitor_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(eval_task, monitor_task, return_exceptions=True)


async def _monitor_vllm_endpoint(health_config: VllmHealthCheckConfig) -> None:
    failures = 0
    last_error = "unknown error"
    while True:
        ok, detail = await _probe_vllm_endpoint(health_config)
        if ok:
            if failures:
                logger.info("vLLM health check recovered for %s", _safe_endpoint_label(health_config.api_base_url))
            failures = 0
        else:
            failures += 1
            last_error = detail
            logger.warning(
                "vLLM health check failed for %s (%d/%d): %s",
                _safe_endpoint_label(health_config.api_base_url),
                failures,
                health_config.failure_threshold,
                detail,
            )
            if failures >= health_config.failure_threshold:
                raise VllmHealthCheckError(
                    "vLLM health check failed "
                    f"{health_config.failure_threshold} consecutive time(s) for "
                    f"{_safe_endpoint_label(health_config.api_base_url)} model={health_config.model}: {last_error}"
                )
        await asyncio.sleep(health_config.interval_seconds)


async def _probe_vllm_endpoint(health_config: VllmHealthCheckConfig) -> tuple[bool, str]:
    health_ok, health_detail = await _probe_vllm_health_endpoint(health_config)
    if health_ok:
        return True, health_detail

    url = _join_v1_path(health_config.api_base_url, "chat/completions")
    headers = dict(health_config.extra_headers or {})
    api_key = os.getenv(health_config.api_key_var or "") if health_config.api_key_var else None
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": health_config.model,
        "messages": [{"role": "user", "content": "health check"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(health_config.timeout_seconds)) as client:
            response = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        return False, f"/health fallback failed ({health_detail}); chat probe {type(exc).__name__}: {exc}"
    if response.status_code >= 500:
        return False, f"/health fallback failed ({health_detail}); chat probe HTTP {response.status_code}: {_response_excerpt(response)}"
    if response.status_code >= 400:
        return False, f"/health fallback failed ({health_detail}); chat probe HTTP {response.status_code}: {_response_excerpt(response)}"
    return True, f"/health fallback failed ({health_detail}); chat probe HTTP {response.status_code}"


async def _probe_vllm_health_endpoint(health_config: VllmHealthCheckConfig) -> tuple[bool, str]:
    url = _join_origin_path(health_config.api_base_url, "health")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(health_config.timeout_seconds)) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        return False, f"/health {type(exc).__name__}: {exc}"
    if response.status_code == 200:
        return True, "/health HTTP 200"
    return False, f"/health HTTP {response.status_code}: {_response_excerpt(response)}"


def _join_v1_path(api_base_url: str, path: str) -> str:
    return f"{api_base_url.rstrip('/')}/{path.lstrip('/')}"


def _join_origin_path(api_base_url: str, path: str) -> str:
    parsed = urlparse(api_base_url)
    if not parsed.scheme or not parsed.netloc:
        return _join_v1_path(api_base_url, path)
    return f"{parsed.scheme}://{parsed.netloc}/{path.lstrip('/')}"


def _is_loopback_url(api_base_url: str) -> bool:
    host = (urlparse(api_base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _safe_endpoint_label(api_base_url: str) -> str:
    parsed = urlparse(api_base_url)
    if not parsed.netloc:
        return api_base_url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _response_excerpt(response: httpx.Response, *, limit: int = 300) -> str:
    text = response.text.strip().replace("\n", " ")
    return text[:limit] if text else response.reason_phrase
