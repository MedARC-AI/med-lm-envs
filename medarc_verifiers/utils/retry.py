import asyncio
import logging
from typing import Any, Awaitable, TypeVar

import httpx
from openai import BadRequestError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from typing_extensions import Protocol
from verifiers.envs.environment import Environment

ModelResponse = Completion | ChatCompletion | None

T = TypeVar("T")


class _AsyncCallable(Protocol[T]):
    def __call__(self) -> Awaitable[T]: ...


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def should_retry_exception(exc: BaseException) -> tuple[bool, str | None]:
    """Identify retryable exceptions from model calls."""
    if isinstance(exc, AssertionError):
        message = str(exc)
        if "Response should always have one choice" in message:
            return True, message
    if isinstance(exc, (BadRequestError, httpx.HTTPStatusError)):
        if _status_code(exc) == 400:
            return True, "HTTP 400 during model call"
    return False, None


def _choices_length(response: Any) -> int | None:
    if hasattr(response, "choices"):
        try:
            choices = response.choices  # type: ignore[assignment]
            return len(choices)  # type: ignore[arg-type]
        except Exception:
            return None
    return None


def should_retry_response(response: ModelResponse) -> tuple[bool, str | None]:
    """Identify retryable model responses (e.g., empty choices)."""
    if response is None:
        return False, None
    choices_len = _choices_length(response)
    if choices_len is None:
        return False, None
    if choices_len != 1:
        return True, f"Unexpected choices len={choices_len}"
    return False, None


async def call_with_retries(
    func: _AsyncCallable[T],
    *,
    attempts: int = 3,
    backoff_s: float = 1.0,
    logger: logging.Logger | None = None,
) -> T:
    """Call an async function with retry handling for known transient issues."""
    log = logger or logging.getLogger(__name__)
    last_exc: BaseException | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            result = await func()
        except Exception as exc:  # noqa: BLE001
            retry, reason = should_retry_exception(exc)
            if retry and attempt < attempts:
                log.warning(
                    "Retryable error on model call (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    reason or exc,
                )
                last_exc = exc
                await asyncio.sleep(backoff_s)
                continue
            raise

        retry, reason = should_retry_response(result)  # type: ignore[arg-type]
        if retry:
            if attempt < attempts:
                log.warning(
                    "Retryable bad response on model call (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    reason,
                )
                await asyncio.sleep(backoff_s)
                continue
            raise RuntimeError(
                f"Retryable bad response persisted after {attempts} attempt(s): {reason}"
            )
        return result
    if last_exc:
        raise last_exc
    raise RuntimeError("call_with_retries exhausted attempts without a result")


def patch_verifiers_model_response_retry(
    *,
    attempts: int = 3,
    backoff_s: float = 1.0,
    logger: logging.Logger | None = None,
) -> None:
    """Monkeypatch Environment.get_model_response to add per-call retries."""
    if getattr(Environment, "_medarc_retry_patched", False):
        return
    log = logger or logging.getLogger("medarc_verifiers.retry")
    original = Environment.get_model_response

    async def _patched_get_model_response(
        self: Environment, *args: Any, **kwargs: Any
    ) -> ModelResponse:
        async def _invoke() -> ModelResponse:
            return await original(self, *args, **kwargs)

        return await call_with_retries(
            _invoke,
            attempts=attempts,
            backoff_s=backoff_s,
            logger=log,
        )

    Environment.get_model_response = _patched_get_model_response  # type: ignore[assignment]
    Environment._medarc_retry_patched = True  # type: ignore[attr-defined]
