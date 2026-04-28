"""Deterministic eval identity helpers for the TOML bench wrapper."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MEDARC_CONFIG_FINGERPRINT_KEY = "medarc_config_fingerprint"
MEDARC_CONFIG_FINGERPRINT_PAYLOAD_KEY = "medarc_config_fingerprint_payload"
MEDARC_VARIANT_ID_KEY = "variant_id"
MEDARC_VARIANT_PAYLOAD_KEY = "variant_payload"

_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SEGMENT_LENGTH = 80
_MAX_VARIANT_ID_LENGTH = 160

_SEMANTIC_SAMPLING_KEYS = {
    "frequency_penalty",
    "logit_bias",
    "max_completion_tokens",
    "max_tokens",
    "min_p",
    "n",
    "presence_penalty",
    "repetition_penalty",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
}
_EXCLUDED_SAMPLING_KEYS = {
    "api_base_url",
    "api_key",
    "api_key_var",
    "base_url",
    "extra_headers",
    "headers",
    "max_retries",
    "metadata",
    "request_timeout",
    "stream",
    "timeout",
}
_EXCLUDED_EXTRA_BODY_KEYS = {
    "metadata",
    "provider",
    "usage",
}


class UnclassifiedSamplingArgError(ValueError):
    """Raised when fingerprinting sees a sampling arg without a policy."""


@dataclass(frozen=True)
class EvalIdentity:
    """Resolved model/env identity plus optional variant metadata."""

    model_id: str
    env_id: str
    variant_id: str | None = None
    variant_payload: dict[str, Any] | None = None

    @property
    def dataset_id(self) -> str:
        if self.variant_id is None:
            return self.env_id
        return f"{self.env_id}::{self.variant_id}"


@dataclass(frozen=True)
class EvalPathPlan:
    """Deterministic result location for one eval config."""

    identity: EvalIdentity
    results_path: Path


def slug_component(value: Any, *, max_length: int = _MAX_SEGMENT_LENGTH) -> str:
    """Return a path-safe slug for one path component."""

    slug = _SLUG_PATTERN.sub("-", str(value).strip()).strip("-._")
    if not slug:
        slug = "value"
    if len(slug) <= max_length:
        return slug
    digest = short_fingerprint(str(value), length=10)
    return f"{slug[: max_length - 11].rstrip('-._')}-{digest}"


def plan_eval_paths(raw_configs: Sequence[Mapping[str, Any]], *, output_root: str | Path) -> list[EvalPathPlan]:
    """Plan deterministic output paths, adding variants for colliding model/env pairs."""

    keys = [(_model_id(config), _env_id(config)) for config in raw_configs]
    counts = Counter(keys)
    semantic_payloads = [_semantic_variant_source(config) for config in raw_configs]
    differing_fields = _differing_fields_by_key(semantic_payloads, keys)

    plans: list[EvalPathPlan] = []
    for idx, (config, key) in enumerate(zip(raw_configs, keys)):
        model_id, env_id = key
        variant_payload: dict[str, Any] | None = None
        variant_id: str | None = None
        if counts[key] > 1:
            variant_payload = extract_variant_payload(semantic_payloads[idx], differing_fields[key])
            variant_id = generate_variant_id(variant_payload)

        identity = EvalIdentity(
            model_id=model_id, env_id=env_id, variant_id=variant_id, variant_payload=variant_payload
        )
        path = Path(output_root) / slug_component(model_id) / slug_component(env_id)
        if variant_id is not None:
            path = path / slug_component(variant_id, max_length=_MAX_VARIANT_ID_LENGTH)
        plans.append(EvalPathPlan(identity=identity, results_path=path))

    _ensure_unique_paths(plans)
    return plans


def extract_variant_payload(config: Mapping[str, Any], field_names: Sequence[str]) -> dict[str, Any]:
    """Return the subset of config fields that distinguishes a variant."""

    payload: dict[str, Any] = {}
    for field_name in field_names:
        if "." in field_name:
            root, nested_key = field_name.split(".", 1)
            value = config.get(root)
            if isinstance(value, Mapping):
                nested_payload = payload.setdefault(root, {})
                if isinstance(nested_payload, dict) and nested_key in value:
                    nested_payload[nested_key] = _canonicalize(value[nested_key])
            else:
                payload.setdefault(root, {})
            continue
        if field_name in config:
            payload[field_name] = _canonicalize(config[field_name])
    return payload


def generate_variant_id(payload: Mapping[str, Any]) -> str:
    """Generate a stable human-readable variant ID from distinguishing fields."""

    if not payload:
        return f"variant-{short_fingerprint(payload)}"

    segments: list[str] = []
    for key, value in sorted(payload.items()):
        if isinstance(value, Mapping):
            for nested_key, nested_value in sorted(value.items()):
                segments.append(_variant_segment(f"{key}.{nested_key}", nested_value))
        else:
            segments.append(_variant_segment(key, value))

    if not segments:
        return "baseline"

    variant_id = "__".join(segments)
    if len(variant_id) <= _MAX_VARIANT_ID_LENGTH and all(not segment.endswith("-hash") for segment in segments):
        return variant_id
    return f"{variant_id[:120].rstrip('-._')}__{short_fingerprint(payload, length=12)}"


def build_fingerprint_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the narrow semantic payload used for config-safe resume checks."""

    payload: dict[str, Any] = {
        "env_args": _canonicalize(config.get("env_args", {})),
        "env_id": _env_id(config),
        "model": _model_id(config),
        "num_examples": config.get("num_examples"),
        "rollouts_per_example": config.get("rollouts_per_example"),
        "sampling_args": normalize_semantic_sampling_args(_sampling_args_with_top_level(config)),
    }
    return payload


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return the stable fingerprint for an eval config's benchmark identity."""

    return short_fingerprint(build_fingerprint_payload(config), length=32)


def normalize_semantic_sampling_args(sampling_args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize provider-independent generation semantics for fingerprinting."""

    if not sampling_args:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in sampling_args.items():
        if key == "extra_body":
            _merge_extra_body_semantics(normalized, value)
        elif key == "reasoning_effort":
            normalized["reasoning_effort"] = _canonicalize(value)
        elif key == "reasoning":
            effort = _extract_reasoning_effort(value)
            if effort is not None:
                normalized["reasoning_effort"] = _canonicalize(effort)
        elif key in _SEMANTIC_SAMPLING_KEYS:
            normalized[key] = _canonicalize(value)
        elif key in _EXCLUDED_SAMPLING_KEYS:
            continue
        else:
            raise UnclassifiedSamplingArgError(f"Sampling arg '{key}' is not classified for resume fingerprinting.")

    return dict(sorted(normalized.items()))


def metadata_identity_fields(config: Mapping[str, Any], identity: EvalIdentity) -> dict[str, Any]:
    """Return MedARC metadata fields to write alongside upstream metadata."""

    payload = build_fingerprint_payload(config)
    return {
        MEDARC_CONFIG_FINGERPRINT_KEY: short_fingerprint(payload, length=32),
        MEDARC_CONFIG_FINGERPRINT_PAYLOAD_KEY: payload,
        MEDARC_VARIANT_ID_KEY: identity.variant_id,
        MEDARC_VARIANT_PAYLOAD_KEY: identity.variant_payload,
    }


def short_fingerprint(value: Any, *, length: int = 12) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _semantic_variant_source(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "env_args": _canonicalize(config.get("env_args", {})),
        "num_examples": config.get("num_examples"),
        "rollouts_per_example": config.get("rollouts_per_example"),
        "sampling_args": normalize_semantic_sampling_args(_sampling_args_with_top_level(config)),
    }


def _sampling_args_with_top_level(config: Mapping[str, Any]) -> dict[str, Any]:
    sampling_args = dict(config.get("sampling_args", {}) or {})
    for key in ("max_tokens", "temperature"):
        if key in config and key not in sampling_args:
            sampling_args[key] = config[key]
    return sampling_args


def _differing_fields_by_key(
    semantic_payloads: Sequence[Mapping[str, Any]], keys: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for payload, key in zip(semantic_payloads, keys):
        grouped[key].append(payload)

    differing: dict[tuple[str, str], list[str]] = {}
    for key, configs in grouped.items():
        if len(configs) < 2:
            differing[key] = []
            continue
        field_names = sorted(set().union(*(payload.keys() for payload in configs)))
        differing[key] = []
        for field_name in field_names:
            values = [payload.get(field_name) for payload in configs]
            if all(isinstance(value, Mapping) for value in values if value is not None):
                nested_names = sorted(
                    {
                        str(nested_key)
                        for value in values
                        if isinstance(value, Mapping)
                        for nested_key in value.keys()
                    }
                )
                differing[key].extend(
                    f"{field_name}.{nested_name}"
                    for nested_name in nested_names
                    if len(
                        {
                            _canonical_json(value.get(nested_name) if isinstance(value, Mapping) else None)
                            for value in values
                        }
                    )
                    > 1
                )
                continue
            if len({_canonical_json(value) for value in values}) > 1:
                differing[key].append(field_name)
    return differing


def _ensure_unique_paths(plans: Sequence[EvalPathPlan]) -> None:
    paths = [plan.results_path for plan in plans]
    duplicate_paths = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicate_paths:
        rendered = ", ".join(str(path) for path in duplicate_paths)
        raise ValueError(f"Deterministic eval path collision after variant planning: {rendered}")


def _variant_segment(key: str, value: Any) -> str:
    key_slug = slug_component(key, max_length=40)
    value_slug = slug_component(_variant_value_text(value), max_length=80)
    if isinstance(value, Mapping | Sequence) and not isinstance(value, str | bytes | bytearray):
        return f"{key_slug}-{value_slug}-{short_fingerprint(value, length=8)}"
    return f"{key_slug}-{value_slug}"


def _variant_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "none"
    if isinstance(value, int | float | str):
        return str(value)
    return "hash"


def _merge_extra_body_semantics(normalized: dict[str, Any], extra_body: Any) -> None:
    if not isinstance(extra_body, Mapping):
        raise UnclassifiedSamplingArgError("sampling_args.extra_body must be a mapping for resume fingerprinting.")

    for key, value in extra_body.items():
        if key == "reasoning":
            effort = _extract_reasoning_effort(value)
            if effort is not None:
                normalized["reasoning_effort"] = _canonicalize(effort)
        elif key in _SEMANTIC_SAMPLING_KEYS:
            normalized[key] = _canonicalize(value)
        elif key in _EXCLUDED_EXTRA_BODY_KEYS or key in _EXCLUDED_SAMPLING_KEYS:
            continue
        else:
            raise UnclassifiedSamplingArgError(
                f"Sampling arg 'extra_body.{key}' is not classified for resume fingerprinting."
            )


def _extract_reasoning_effort(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get("effort") or value.get("reasoning_effort")


def _model_id(config: Mapping[str, Any]) -> str:
    value = config.get("model")
    if not value:
        raise ValueError(
            "Eval config must include resolved 'model' for deterministic identity; build EvalConfig before planning paths."
        )
    return str(value)


def _env_id(config: Mapping[str, Any]) -> str:
    value = config.get("env_id")
    if not value:
        raise ValueError("Eval config must include 'env_id' for deterministic identity.")
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), default=str)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return [_canonicalize(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "EvalIdentity",
    "EvalPathPlan",
    "MEDARC_CONFIG_FINGERPRINT_KEY",
    "MEDARC_CONFIG_FINGERPRINT_PAYLOAD_KEY",
    "MEDARC_VARIANT_ID_KEY",
    "MEDARC_VARIANT_PAYLOAD_KEY",
    "UnclassifiedSamplingArgError",
    "build_fingerprint_payload",
    "config_fingerprint",
    "extract_variant_payload",
    "generate_variant_id",
    "metadata_identity_fields",
    "normalize_semantic_sampling_args",
    "plan_eval_paths",
    "short_fingerprint",
    "slug_component",
]
