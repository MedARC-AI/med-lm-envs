import httpx
import pytest
from verifiers.errors import ModelError

from medarc_verifiers.utils.retry import call_with_retries, should_retry_exception


class DummyResponse:
    def __init__(self, choices):
        self.choices = choices


@pytest.mark.asyncio
async def test_call_with_retries_recovers_from_http_400():
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(400, request=request)
            raise httpx.HTTPStatusError("bad request", request=request, response=response)
        return "ok"

    result = await call_with_retries(flaky, attempts=2, backoff_s=0)
    assert result == "ok"
    assert attempts == 2


@pytest.mark.asyncio
async def test_call_with_retries_recovers_from_empty_choices_response():
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return DummyResponse([])
        return DummyResponse([{"text": "hello"}])

    result = await call_with_retries(flaky, attempts=2, backoff_s=0)
    assert isinstance(result, DummyResponse)
    assert len(result.choices) == 1
    assert attempts == 2


@pytest.mark.asyncio
async def test_call_with_retries_raises_on_non_retryable_error():
    async def bad():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await call_with_retries(bad, attempts=3, backoff_s=0)


class DummyPolicyViolation403(Exception):
    def __init__(self):
        self.status_code = 403
        self.body = {"error": {"type": "policy_violation", "code": "content_moderation"}}
        super().__init__("Error code: 403 - policy_violation")


@pytest.mark.asyncio
async def test_call_with_retries_recovers_from_403_policy_violation_with_one_retry():
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DummyPolicyViolation403()
        return "ok"

    result = await call_with_retries(flaky, attempts=5, backoff_s=0)
    assert result == "ok"
    assert attempts == 2


@pytest.mark.asyncio
async def test_call_with_retries_403_policy_violation_only_retries_once():
    attempts = 0

    async def always_bad():
        nonlocal attempts
        attempts += 1
        raise DummyPolicyViolation403()

    with pytest.raises(DummyPolicyViolation403):
        await call_with_retries(always_bad, attempts=5, backoff_s=0)
    assert attempts == 2


@pytest.mark.asyncio
async def test_call_with_retries_unwraps_verifiers_model_error_chain():
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(500, request=request)
            try:
                raise httpx.HTTPStatusError("server error", request=request, response=response)
            except httpx.HTTPStatusError as exc:
                raise ModelError("wrapped model error") from exc
        return "ok"

    result = await call_with_retries(flaky, attempts=2, backoff_s=0)
    assert result == "ok"
    assert attempts == 2


def test_should_retry_exception_unwraps_verifiers_model_error_chain():
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)
    try:
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)
    except httpx.HTTPStatusError as exc:
        wrapped = ModelError("wrapped model error")
        wrapped.__cause__ = exc

    retry, code, _reason, _delay = should_retry_exception(wrapped)
    assert retry is True
    assert code == 429


def test_should_retry_exception_unwraps_verifiers_model_error_with_wrapper_layer():
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)
    try:
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)
    except httpx.HTTPStatusError as inner_exc:
        try:
            raise RuntimeError("intermediate wrapper") from inner_exc
        except RuntimeError as wrapper_exc:
            wrapped = ModelError("wrapped model error")
            wrapped.__cause__ = wrapper_exc

    retry, code, _reason, _delay = should_retry_exception(wrapped)
    assert retry is True
    assert code == 429
