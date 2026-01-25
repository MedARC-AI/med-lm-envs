"""Helpers for parsing CLI-provided override arguments.

This module handles the parsing and merging of CLI override flags:
- --env-args / --sampling-args (JSON payload)
- --env-arg / --sampling-arg (KEY=VALUE pairs, repeatable)

Precedence: KEY=VALUE pairs override JSON payload values when both are provided.
"""

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
    """Merge JSON and KEY=VALUE override inputs.

    Parses both JSON object flags (e.g., --env-args '{"key": "value"}') and
    repeatable KEY=VALUE pair flags (e.g., --env-arg key=value), with pairs
    taking precedence over JSON values.

    Args:
        json_payload: JSON string to parse (from --*-args flag)
        pairs: List of KEY=VALUE strings (from repeated --*-arg flags)
        json_flag: Name of JSON flag for error messages (e.g., "--env-args")
        pair_flag: Name of pair flag for error messages (e.g., "--env-arg")

    Returns:
        Merged dictionary, or None if no overrides provided.

    Example:
        json_payload = '{"seed": 42, "workers": 4}'
        pairs = ["seed=123", "verbose=true"]
        → Result: {"seed": 123, "workers": 4, "verbose": True}
          (pairs override JSON, with smart type coercion)
    """
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
    """Smart type coercion for CLI KEY=VALUE inputs.

    Attempts to parse values in this order:
    1. Boolean literals: "true"/"false" → bool
    2. None literals: "null"/"none" → None
    3. JSON objects/arrays: starts with "{" or "[" → parsed JSON
    4. Integers: "123" → int
    5. Floats: "1.5" → float
    6. Strings: everything else → str

    This provides better UX than requiring JSON for everything.

    Examples:
        "true" → True
        "false" → False
        "123" → 123
        "1.5" → 1.5
        "null" → None
        '{"a": 1}' → {"a": 1}
        "hello" → "hello"
    """
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
