from dataclasses import dataclass, field
import os
from typing import Dict, Iterable, Optional, Tuple, Union
from .sampling_args import sanitize_sampling_args_for_openai


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
    name: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    segments: Tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", _split_segments(_normalize_judge_name(self.name)))

    def as_dict(self) -> Dict[str, Union[float, int, str]]:
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
        payload["extra_body"] = {"usage": {"include": True}}
        return payload


# These defaults follow the model provider's recommended settings,
# with the excpetion of Claude and GPT-4o and GPT-4.1 which follow OpenAI's HealthBench judge settings.
_JUDGE_DEFAULTS: Iterable[JudgeSamplingDefaults] = (
    JudgeSamplingDefaults(
        name="claude-4.0",
        temperature=0.5,
    ),
    JudgeSamplingDefaults(
        name="claude-4.5",
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
        name="glm-4.6",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
    ),
    JudgeSamplingDefaults(
        name="glm-4.5",
        temperature=0.6,
        top_p=0.95,
    ),
    JudgeSamplingDefaults(
        name="gpt-4.1",
        temperature=0.5,
        top_p=1.0,
    ),
    JudgeSamplingDefaults(
        name="gpt-4o",
        temperature=0.5,
        top_p=1.0,
    ),
    JudgeSamplingDefaults(
        name="gpt-5",
        reasoning_effort="low",  # TODO: support responses api reasoning effort when verifiers adds suport for it
    ),
    JudgeSamplingDefaults(
        name="gpt-5.1",
        reasoning_effort="low",  # TODO: support responses api reasoning effort when verifiers adds suport for it
    ),
    JudgeSamplingDefaults(
        name="gpt-oss",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        reasoning_effort="medium",  # TODO: support responses api reasoning effort when verifiers adds suport for it
    ),
    JudgeSamplingDefaults(
        name="grok-4",
        temperature=0.2,
    ),
    JudgeSamplingDefaults(
        name="kimi-k2-thinking",
        temperature=1.0,
    ),
    JudgeSamplingDefaults(
        name="qwen-3-thinking",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    ),
)


def judge_sampling_args_and_headers(
    judge_name: str, base_url: str | None = None, timeout: int | None = 300
) -> Tuple[Dict[str, Union[float, int, str]], Optional[Dict[str, str]]]:
    """Return the sampling defaults for the provided judge name.

    Matches are case-insensitive and use the canonical judge identifier.
    """

    normalized = _normalize_judge_name(judge_name)
    candidate_segments = _split_segments(normalized)
    for judge_defaults in _JUDGE_DEFAULTS:
        if _is_subsequence(judge_defaults.segments, candidate_segments):
            if base_url == "https://api.pinference.ai/api/v1" and os.environ.get("PRIME_TEAM_ID") is not None:
                prime_team_id = {"X-Prime-Team-ID": os.environ.get("PRIME_TEAM_ID")}
            else:
                prime_team_id = None
            payload = judge_defaults.as_dict()
            if timeout is not None:
                payload["timeout"] = timeout
            return sanitize_sampling_args_for_openai(payload), prime_team_id

    raise KeyError(f"No sampling defaults available for judge “{judge_name}”.")


def default_judge_api_key(base_url: str | None = None) -> str | None:
    if base_url == "https://api.pinference.ai/api/v1" and os.environ.get("PRIME_API_KEY") is not None:
        return os.environ.get("PRIME_API_KEY")
    elif os.environ.get("OPENAI_API_KEY") is not None:
        return os.environ.get("OPENAI_API_KEY")
    elif os.environ.get("JUDGE_API_KEY") is not None:
        return os.environ.get("JUDGE_API_KEY")
    else:
        return None
