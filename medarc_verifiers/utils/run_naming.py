"""Shared helpers for generating MedARC run identifiers."""

from __future__ import annotations

from datetime import UTC, datetime

from medarc_verifiers.cli.utils.shared import slugify


def generate_run_id(name: str | None) -> str:
    base = slugify(name or "run")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{base}-{timestamp}"


__all__ = ["generate_run_id"]
