"""Core data structures for the unified CLI implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycles resolved in later phases
    from ._schemas import EnvironmentConfigSchema, ModelConfigSchema  # noqa: F401
    from ._manifest import RunManifest  # noqa: F401


@dataclass(slots=True)
class EvalJob:
    """Unified representation of an evaluation job."""

    job_id: str
    model: Any  # ModelConfigSchema in practice
    env: Any  # EnvironmentConfigSchema in practice
    overrides: dict[str, Any]


@dataclass(slots=True)
class EvalRun:
    """Collection of jobs that share a manifest/output destination."""

    run_id: str
    jobs: list[EvalJob]
    output_dir: Path
    manifest: Any | None = None  # RunManifest in practice


__all__ = ["EvalJob", "EvalRun"]
