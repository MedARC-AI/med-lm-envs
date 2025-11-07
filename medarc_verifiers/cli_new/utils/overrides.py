"""Helpers for parsing CLI-provided override arguments."""

from __future__ import annotations

import json
from typing import Any, Sequence


def build_cli_override(
    *,
    json_payload: str | None,
    pairs: Sequence[str] | None,
    json_flag: str,
    pair_flag: str,
) -> dict[str, Any] | None:
    """Merge JSON and KEY=VALUE override inputs."""
    json_args = _parse_json_mapping(json_payload, flag=json_flag)
    pair_args = _parse_key_value_pairs(pairs, flag=pair_flag)

    if not json_args and not pair_args:
        return None

    merged = dict(json_args)
    merged.update(pair_args)
    return merged


def _parse_json_mapping(value: str | None, *, flag: str) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - argparse messaging
        raise ValueError(f"{flag} must be valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{flag} must be a JSON object.")
    return decoded


def _parse_key_value_pairs(values: Sequence[str] | None, *, flag: str) -> dict[str, Any]:
    if not values:
        return {}
    parsed: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{flag} entries must use the form KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{flag} entries must include a key before '='.")
        parsed[key] = _coerce_cli_value(value.strip())
    return parsed


def _coerce_cli_value(raw: str) -> Any:
    if not raw:
        return ""
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


__all__ = ["build_cli_override"]
