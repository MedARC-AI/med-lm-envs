"""Deterministic eval identity helpers for the TOML bench wrapper."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MEDARC_VARIANT_ID_KEY = "variant_id"
BASE_VARIANT_ID = "base"

_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SEGMENT_LENGTH = 80
_MAX_VARIANT_ID_LENGTH = 160


@dataclass(frozen=True)
class EvalIdentity:
    """Resolved model/env identity plus semantic variant metadata."""

    model_id: str
    env_id: str
    variant_id: str = BASE_VARIANT_ID

    @property
    def dataset_id(self) -> str:
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
    digest = _short_text_digest(str(value), length=10)
    return f"{slug[: max_length - 11].rstrip('-._')}-{digest}"


def generate_variant_id(payload: Mapping[str, Any]) -> str:
    """Generate a stable human-readable variant id for legacy export config keys."""

    if not payload:
        return BASE_VARIANT_ID

    segments: list[str] = []
    for key, value in sorted(payload.items()):
        if isinstance(value, Mapping):
            for nested_key, nested_value in sorted(value.items()):
                segments.append(_variant_segment(f"{key}.{nested_key}", nested_value))
        else:
            segments.append(_variant_segment(key, value))

    if not segments:
        return BASE_VARIANT_ID

    variant_id = "__".join(segments)
    if len(variant_id) <= _MAX_VARIANT_ID_LENGTH and all(not segment.endswith("-hash") for segment in segments):
        return variant_id
    return f"{variant_id[:120].rstrip('-._')}__{_short_json_digest(payload, length=12)}"


def plan_eval_paths(raw_configs: Sequence[Mapping[str, Any]], *, output_root: str | Path) -> list[EvalPathPlan]:
    """Plan deterministic output paths for TOML bench eval configs."""

    keys = [(_model_id(config), _env_id(config)) for config in raw_configs]
    plans: list[EvalPathPlan] = []
    for idx, (config, key) in enumerate(zip(raw_configs, keys)):
        model_id, env_id = key
        variant_id = _variant_id(config, index=idx + 1)

        identity = EvalIdentity(model_id=model_id, env_id=env_id, variant_id=variant_id)
        path = Path(output_root) / slug_component(model_id) / slug_component(env_id) / variant_id
        plans.append(EvalPathPlan(identity=identity, results_path=path))

    _ensure_unique_identities(plans)
    _ensure_unique_slugs(plans)
    return plans


def _ensure_unique_identities(plans: Sequence[EvalPathPlan]) -> None:
    identities = [(plan.identity.model_id, plan.identity.env_id, plan.identity.variant_id) for plan in plans]
    duplicates = sorted(identity for identity, count in Counter(identities).items() if count > 1)
    if duplicates:
        rendered = ", ".join(
            f"model={model!r}, env_id={env_id!r}, variant_id={variant_id!r}"
            for model, env_id, variant_id in duplicates
        )
        raise ValueError(f"Duplicate TOML eval identity; add a distinct variant_id/name: {rendered}")


def _ensure_unique_slugs(plans: Sequence[EvalPathPlan]) -> None:
    _raise_slug_collisions(
        "model",
        ((slug_component(plan.identity.model_id), plan.identity.model_id) for plan in plans),
    )

    _raise_slug_collisions(
        "env",
        (
            (
                f"{slug_component(plan.identity.model_id)}/{slug_component(plan.identity.env_id)}",
                f"{plan.identity.model_id}/{plan.identity.env_id}",
            )
            for plan in plans
        ),
    )
    _raise_slug_collisions(
        "variant",
        (
            (
                "/".join(
                    (
                        slug_component(plan.identity.model_id),
                        slug_component(plan.identity.env_id),
                        slug_component(plan.identity.variant_id, max_length=_MAX_VARIANT_ID_LENGTH),
                    )
                ),
                f"{plan.identity.model_id}/{plan.identity.env_id}/{plan.identity.variant_id}",
            )
            for plan in plans
        ),
    )

    paths = [plan.results_path for plan in plans]
    duplicate_paths = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicate_paths:
        rendered = ", ".join(str(path) for path in duplicate_paths)
        raise ValueError(f"Deterministic eval path collision: {rendered}")


def _raise_slug_collisions(label: str, pairs: Iterable[tuple[str, str]]) -> None:
    values_by_slug: dict[str, set[str]] = {}
    for slug, value in pairs:
        values_by_slug.setdefault(slug, set()).add(value)
    collisions = {slug: sorted(values) for slug, values in values_by_slug.items() if len(values) > 1}
    if not collisions:
        return
    rendered = "; ".join(f"{slug}: {values}" for slug, values in sorted(collisions.items()))
    raise ValueError(f"Deterministic eval {label} slug collision: {rendered}")


def _variant_id(config: Mapping[str, Any], *, index: int) -> str:
    raw_variant = config.get("variant_id")
    raw_name = config.get("name")
    variant = _normalize_variant(raw_variant, config=config, field="variant_id", index=index)
    name = _normalize_variant(raw_name, config=config, field="name", index=index)
    if variant and name and variant != name:
        raise ValueError(
            f"TOML eval {index} has conflicting variant_id/name values: {variant!r} != {name!r}."
        )
    return variant or name or BASE_VARIANT_ID


def _normalize_variant(value: Any, *, config: Mapping[str, Any], field: str, index: int) -> str | None:
    if value is None:
        return None
    text = _expand_variant_template(str(value).strip(), config)
    if not text:
        raise ValueError(f"TOML eval {index} {field} must not be empty.")
    if slug_component(text, max_length=_MAX_VARIANT_ID_LENGTH) != text:
        raise ValueError(
            f'TOML eval {index} {field} {text!r} is not path-safe. '
            'Use only letters, numbers, ".", "_", and "-", for example "shuffle_seed-1618".'
        )
    return text


def _expand_variant_template(template: str, config: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(1).strip()
        value: Any = config
        for part in path.split("."):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            else:
                raise ValueError(f"Variant template references unknown field: {path}")
        return str(value)

    return re.sub(r"\{([^{}]+)\}", replace, template).strip()


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


def _short_text_digest(value: str, *, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _short_json_digest(value: Any, *, length: int) -> str:
    encoded = json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _variant_segment(key: str, value: Any) -> str:
    key_slug = slug_component(key, max_length=40)
    value_slug = slug_component(_variant_value_text(value), max_length=80)
    if isinstance(value, Mapping | Sequence) and not isinstance(value, str | bytes | bytearray):
        return f"{key_slug}-{value_slug}-{_short_json_digest(value, length=8)}"
    return f"{key_slug}-{value_slug}"


def _variant_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "none"
    if isinstance(value, int | float | str):
        return str(value)
    return "hash"


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
    "BASE_VARIANT_ID",
    "EvalIdentity",
    "EvalPathPlan",
    "MEDARC_VARIANT_ID_KEY",
    "generate_variant_id",
    "plan_eval_paths",
    "slug_component",
]
