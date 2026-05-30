from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli import bench_health


def _eval_config(*, url: str, client_type: str = "openai_chat_completions") -> SimpleNamespace:
    return SimpleNamespace(
        model="served-model",
        client_config=SimpleNamespace(
            api_base_url=url,
            api_key_var="VLLM_API_KEY",
            client_type=client_type,
            extra_headers={"X-Test": "1"},
        ),
    )


def test_auto_health_check_enables_for_local_vllm_provider() -> None:
    config = bench_health.resolve_vllm_health_check_config(
        _eval_config(url="http://127.0.0.1:8000/v1"),
        provider="local",
        mode="auto",
        interval_seconds=600,
        timeout_seconds=120,
        failure_threshold=2,
    )

    assert config is not None
    assert config.api_base_url == "http://127.0.0.1:8000/v1"
    assert config.timeout_seconds == 120
    assert config.interval_seconds == 600
    assert config.failure_threshold == 2


def test_auto_health_check_skips_hosted_api_url() -> None:
    config = bench_health.resolve_vllm_health_check_config(
        _eval_config(url="https://api.pinference.ai/api/v1"),
        provider="prime",
        mode="auto",
        interval_seconds=600,
        timeout_seconds=120,
        failure_threshold=2,
    )

    assert config is None


def test_health_check_fails_after_two_bad_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    probe_calls = 0

    async def fake_probe(_config):
        nonlocal probe_calls
        probe_calls += 1
        return False, "HTTP 500"

    async def fake_eval():
        await asyncio.Event().wait()

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(bench_health, "_probe_vllm_endpoint", fake_probe)
    monkeypatch.setattr(bench_health.asyncio, "sleep", no_sleep)

    health_config = bench_health.VllmHealthCheckConfig(
        api_base_url="http://127.0.0.1:8000/v1",
        model="served-model",
        interval_seconds=600,
        timeout_seconds=120,
        failure_threshold=2,
    )

    with pytest.raises(bench_health.VllmHealthCheckError):
        asyncio.run(bench_health.run_with_vllm_health_check(fake_eval, health_config))

    assert probe_calls == 2


def test_probe_uses_health_endpoint_before_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, str]] = []

    class FakeResponse:
        status_code = 200
        text = "ok"
        reason_phrase = "OK"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            requests.append(("GET", url))
            return FakeResponse()

        async def post(self, url, headers=None, json=None):
            requests.append(("POST", url))
            return FakeResponse()

    monkeypatch.setattr(bench_health.httpx, "AsyncClient", FakeClient)

    ok, detail = asyncio.run(
        bench_health._probe_vllm_endpoint(
            bench_health.VllmHealthCheckConfig(api_base_url="http://127.0.0.1:8000/v1", model="served-model")
        )
    )

    assert ok is True
    assert detail == "/health HTTP 200"
    assert requests == [("GET", "http://127.0.0.1:8000/health")]


def test_probe_falls_back_to_chat_when_health_is_not_200(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, str]] = []

    class HealthResponse:
        status_code = 503
        text = "starting"
        reason_phrase = "Service Unavailable"

    class ChatResponse:
        status_code = 200
        text = "{}"
        reason_phrase = "OK"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            requests.append(("GET", url))
            return HealthResponse()

        async def post(self, url, headers=None, json=None):
            requests.append(("POST", url))
            return ChatResponse()

    monkeypatch.setattr(bench_health.httpx, "AsyncClient", FakeClient)

    ok, detail = asyncio.run(
        bench_health._probe_vllm_endpoint(
            bench_health.VllmHealthCheckConfig(api_base_url="http://127.0.0.1:8000/v1", model="served-model")
        )
    )

    assert ok is True
    assert "/health fallback failed" in detail
    assert requests == [
        ("GET", "http://127.0.0.1:8000/health"),
        ("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    ]
