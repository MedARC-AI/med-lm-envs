from dataclasses import dataclass, field
import os
from typing import Dict, Iterable, Optional, Tuple, Union
from .sampling_args import sanitize_sampling_args_for_openai
from .prime_inference import PRIME_INFERENCE_URL, _resolve_include_usage


def _normalize_judge_name(name: str) -> str:
    return name.lower().replace("_", "-").replace("/", "-")


def _split_segments(normalized: str) -> Tuple[str, ...]:
    return tuple(segment for segment in normalized.split("-") if segment)


def _is_subsequence(needle: Tuple[str, ...], haystack: Tuple[str, ...]) -> bool:
    if not needle:
        return False

    index = 0
    for segment in haystack:
        if segment == needle[index]:
            index += 1
            if index == len(needle):
                return True
    return False


@dataclass(frozen=True)
class JudgeSamplingDefaults:
    name: str | Tuple[str, ...]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    segments_list: Tuple[Tuple[str, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names: Tuple[str, ...] = (self.name,) if isinstance(self.name, str) else self.name
        object.__setattr__(
            self,
            "segments_list",
            tuple(_split_segments(_normalize_judge_name(name)) for name in names),
        )

    def as_dict(self, *, include_usage: bool = True) -> Dict[str, Union[float, int, str]]:
        payload: Dict[str, Union[float, int, str]] = {}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if include_usage:
            payload["extra_body"] = {"usage": {"include": True}}
        return payload


# These defaults follow the model provider's recommended settings,
# with the excpetion of Claude and GPT-4o and GPT-4.1 which follow OpenAI's HealthBench judge settings.
_JUDGE_DEFAULTS: Iterable[JudgeSamplingDefaults] = (
    JudgeSamplingDefaults(
        name=("claude-4.0", "claude-4.5"),
        temperature=0.5,
    ),
    JudgeSamplingDefaults(
        name="deepseek-r1",
        temperature=0.6,
    ),
    JudgeSamplingDefaults(
        name="gemini-2.5",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    ),
    JudgeSamplingDefaults(
        name="gemini-3",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        reasoning_effort="low",  # TODO: support responses api reasoning effort when verifiers adds suport for it
    ),
    JudgeSamplingDefaults(
        name="glm-4.5",
        temperature=0.6,
        top_p=0.95,
    ),
    JudgeSamplingDefaults(
        name=("glm-4.6", "glm-4.7"),
        temperature=1.0,
        top_p=0.95,
        top_k=40,
    ),
    JudgeSamplingDefaults(
        name=("gpt-4.1", "gpt-4o"),
        temperature=0.5,
        top_p=1.0,
    ),
    JudgeSamplingDefaults(
        name=("gpt-5", "gpt-5.1", "gpt-5.2"),
        reasoning_effort="low",  # TODO: support responses api reasoning effort when verifiers adds suport for it
    ),
    JudgeSamplingDefaults(
        name="gpt-oss",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        reasoning_effort="low",  # TODO: support responses api reasoning effort when verifiers adds suport for it
    ),
    JudgeSamplingDefaults(
        name=("grok-4", "grok-4.1"),
        temperature=0.2,
    ),
    JudgeSamplingDefaults(
        name="kimi-k2-thinking",
        temperature=1.0,
    ),
    JudgeSamplingDefaults(
        name=("qwen-3-thinking", "qwen3-max"),
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    ),
)


def judge_sampling_args_and_headers(
    judge_name: str,
    base_url: str | None = None,
    timeout: int | None = 300,
    include_usage: bool | None = None,
) -> Tuple[Dict[str, Union[float, int, str]], Optional[Dict[str, str]]]:
    """Return the sampling defaults for the provided judge name.

    Matches are case-insensitive and use the canonical judge identifier.

    Args:
        judge_name: The name of the judge model.
        base_url: The base URL for the API. Used for Prime Inference detection.
        timeout: Request timeout in seconds.
        include_usage: Whether to include usage reporting in extra_body.
            If None (default), checks MEDARC_INCLUDE_USAGE env var, then
            auto-detects based on base_url:
            - True if base_url is Prime Inference URL
            - False otherwise
    """

    normalized = _normalize_judge_name(judge_name)
    candidate_segments = _split_segments(normalized)
    for judge_defaults in _JUDGE_DEFAULTS:
        if any(_is_subsequence(segments, candidate_segments) for segments in judge_defaults.segments_list):
            is_prime_inference = base_url == PRIME_INFERENCE_URL
            if is_prime_inference and os.environ.get("PRIME_TEAM_ID") is not None:
                prime_team_id = {"X-Prime-Team-ID": os.environ.get("PRIME_TEAM_ID")}
            else:
                prime_team_id = None

            effective_include_usage = _resolve_include_usage(include_usage, is_prime_inference)

            payload = judge_defaults.as_dict(include_usage=effective_include_usage)
            if timeout is not None:
                payload["timeout"] = timeout
            return sanitize_sampling_args_for_openai(payload), prime_team_id

    raise KeyError(f"No sampling defaults available for judge '{judge_name}'.")


def default_judge_api_key(base_url: str | None = None) -> str | None:
    if base_url == PRIME_INFERENCE_URL and os.environ.get("PRIME_API_KEY") is not None:
        return os.environ.get("PRIME_API_KEY")
    elif os.environ.get("OPENAI_API_KEY") is not None:
        return os.environ.get("OPENAI_API_KEY")
    elif os.environ.get("JUDGE_API_KEY") is not None:
        return os.environ.get("JUDGE_API_KEY")
    else:
        return None
