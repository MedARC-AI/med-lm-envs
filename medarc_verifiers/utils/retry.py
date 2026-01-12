import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Awaitable, TypeVar

import httpx
from openai import BadRequestError, RateLimitError
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


def _parse_retry_delay(value: Any) -> float | None:
    """Parse retry delay values like '35s' or numbers."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            stripped = value.strip().lower()
            if stripped.endswith("s"):
                stripped = stripped[:-1]
            return float(stripped)
        except Exception:
            return None
    return None


def _extract_retry_delay(exc: BaseException) -> float | None:
    """Attempt to extract retry delay seconds from common 429 payloads."""
    # httpx.HTTPStatusError and OpenAI errors usually carry a response with json/text
    resp = getattr(exc, "response", None)
    # Try structured JSON first
    for candidate in (getattr(resp, "json", None), getattr(resp, "text", None)):
        try:
            if callable(candidate):
                payload = candidate()
            else:
                payload = candidate
            if isinstance(payload, str):
                import json

                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = None
            if not payload:
                continue
            # Normalize Gemini-style payloads which can be a dict or a one-element list
            if isinstance(payload, list) and payload:
                # sometimes returned as [{'error': {...}}]
                payload = payload[0]
            # Gemini-style error: {"error": {"details": [{"@type": "...RetryInfo", "retryDelay": "35s"}]}}
            error_block = payload.get("error") if isinstance(payload, dict) else None
            details = error_block.get("details") if isinstance(error_block, dict) else None
            if isinstance(details, list):
                for item in details:
                    if isinstance(item, dict) and "retryDelay" in item:
                        delay = _parse_retry_delay(item.get("retryDelay"))
                        if delay is not None:
                            return delay
        except Exception:
            continue
    # Fallback: sometimes the message contains "Please retry in 35.12s."
    try:
        text = str(exc)
        import re

        match = re.search(r"retry in ([0-9]+(?:\\.[0-9]+)?)s", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception:
        return None
    return None


def should_retry_exception(exc: BaseException) -> tuple[bool, int | None, str | None, float | None]:
    """Identify retryable exceptions from model calls."""
    if isinstance(exc, AssertionError):
        message = str(exc)
        if "Response should always have one choice" in message:
            return True, None, message, None
    status = _status_code(exc)
    retry_delay = _extract_retry_delay(exc) if status == 429 else None
    if isinstance(exc, (BadRequestError, httpx.HTTPStatusError)):
        if status == 400:
            return True, 400, "HTTP 400 during model call", None
        if status == 429:
            return True, 429, f"HTTP 429 rate limited: {retry_delay}", retry_delay
        if status == 500:
            return True, 500, "HTTP 500 internal error", None
    if isinstance(exc, RateLimitError):
        if status == 429:
            return True, 429, f"HTTP 429 too many tokens per minute: {retry_delay}", retry_delay
    if status == 429:
        return True, 429, f"HTTP 429 rate limited: {retry_delay}", retry_delay
    if status == 500:
        return True, 500, "HTTP 500 internal error", None
    return False, None, None, None


def _choices_length(response: Any) -> tuple[int | None, bool]:
    """Return (len, is_none) to distinguish empty vs missing."""
    if hasattr(response, "choices"):
        try:
            choices = response.choices  # type: ignore[assignment]
            if choices is None:
                return None, True
            return len(choices), False  # type: ignore[arg-type]
        except Exception:
            return None, False
    return None, False


def should_retry_response(response: ModelResponse) -> tuple[bool, str | None]:
    """Identify retryable model responses (e.g., empty choices)."""
    if response is None:
        return True, "Response is None"
    choices_len, is_none = _choices_length(response)
    if is_none:
        return True, "Response choices is None"
    if choices_len is None:
        return True, "Response choices missing length"
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
            retry, code, reason, retry_delay = should_retry_exception(exc)
            if retry and attempt < attempts:
                delay = (
                    retry_delay
                    if retry_delay is not None
                    else backoff_s + (random.uniform(1, 10) if code in (429, 500) else 0)
                )
                log.warning(
                    "Retryable error on model call (attempt %d/%d): %s (sleep %.2fs)",
                    attempt,
                    attempts,
                    reason or exc,
                    delay,
                )
                last_exc = exc
                await asyncio.sleep(delay)
                continue
            raise

        if result is None or hasattr(result, "choices"):
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
                raise RuntimeError(f"Retryable bad response persisted after {attempts} attempt(s): {reason}")
        return result
    if last_exc:
        raise last_exc
    raise RuntimeError("call_with_retries exhausted attempts without a result")


def patch_verifiers_model_response_retry(
    *,
    attempts: int = 3,
    backoff_s: float = 1.0,
    log_path: str | Path = "medarc_model_retry.log",
) -> None:
    """Monkeypatch Environment.get_model_response to add per-call retries and log to a file."""
    if getattr(Environment, "_medarc_retry_patched", False):
        return

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("medarc_verifiers.retry")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file for h in log.handlers):  # type: ignore[attr-defined]
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

    # Suppress noisy 429 errors emitted inside Environment.get_model_response;
    # our retry wrapper will log the retry instead.
    suppress_logger = logging.getLogger("verifiers.envs.SingleTurnEnv")
    if not any(getattr(f, "_medarc_429_filter", False) for f in suppress_logger.filters):

        class _429Filter(logging.Filter):
            _medarc_429_filter = True

            def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
                msg = record.getMessage()
                if record.levelno >= logging.ERROR and "Error getting model response" in msg and "429" in msg:
                    return False
                return True

        suppress_logger.addFilter(_429Filter())

    original = Environment.get_model_response

    async def _patched_get_model_response(self: Environment, *args: Any, **kwargs: Any) -> ModelResponse:
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
