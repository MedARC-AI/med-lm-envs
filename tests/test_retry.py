import httpx
import pytest

from medarc_verifiers.utils.retry import call_with_retries


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
