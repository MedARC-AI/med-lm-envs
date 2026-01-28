"""Pydantic schema stubs for the unified CLI configuration system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RESERVED_MATRIX_KEYS = {
    "id",
    "module",
    "env_args",
    "matrix",
    "matrix_exclude",
    "matrix_id_format",
    "matrix_base_id",
    "state_columns",
}


# NOTE: These schema definitions are intentionally incomplete. They provide
# the structural scaffolding required to start wiring the config loader and
# will be expanded in subsequent steps of the integration plan.


class ModelConfigSchema(BaseModel):
    """Schema for model configuration entries (keyed by identifier)."""

    resume_tolerant_fields: ClassVar[set[str]] = frozenset(
        {
            "api_key_var",
            "api_base_url",
            "endpoints_path",
            "headers",
            "timeout",
            "max_connections",
            "max_keepalive_connections",
            "max_retries",
            "max_concurrent",
        }
    )

    id: str | None = Field(
        None,
        description="Optional model identifier (legacy list format).",
    )
    model: str | None = Field(None, description="Provider-specific model slug.")
    headers: list[str] | dict[str, str] | None = Field(
        None,
        description="Optional HTTP headers to attach to requests.",
    )
    sampling_args: dict[str, Any] = Field(default_factory=dict)
    env_args: dict[str, Any] = Field(default_factory=dict)
    env_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    api_key_var: str | None = None
    api_base_url: str | None = None
    endpoints_path: str | None = None
    timeout: float | None = Field(None, ge=0)
    max_connections: int | None = Field(None, ge=1)
    max_keepalive_connections: int | None = Field(None, ge=1)
    max_retries: int | None = Field(None, ge=0)
    max_concurrent: int | None = Field(None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def merge_legacy_params(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        params = data.get("params")
        if not isinstance(params, dict):
            return data
        merged = dict(params)
        for key, value in data.items():
            if key == "params":
                continue
            merged[key] = value
        merged.setdefault("id", data.get("id"))
        return merged

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: list[str] | dict[str, str] | None) -> list[str] | dict[str, str] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, str):
                    msg = "Header entries must be strings when provided as a list."
                    raise ValueError(msg)
        else:
            msg = "Headers must be provided as a list of strings or a mapping."
            raise ValueError(msg)
        return value

    @field_validator("env_args")
    @classmethod
    def default_model_env_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @field_validator("env_overrides", mode="before")
    @classmethod
    def validate_env_overrides(cls, value: Any) -> dict[str, dict[str, Any]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("env_overrides must be a mapping of environment ids to mappings.")
        normalized: dict[str, dict[str, Any]] = {}
        for env_id, override in value.items():
            if not isinstance(env_id, str) or not env_id:
                raise ValueError("env_overrides keys must be non-empty strings.")
            if not isinstance(override, dict):
                raise ValueError(f"env_overrides['{env_id}'] must be a mapping.")
            normalized[env_id] = dict(override)
        return normalized


class EnvironmentExportConfig(BaseModel):
    """Optional export customization embedded in environment configs."""

    model_config = ConfigDict(populate_by_name=True)

    extra_columns: list[str] = Field(default_factory=list, alias="keep_columns")
    drop_columns: list[str] = Field(default_factory=list)
    combine_rollouts: bool = True
    answer_column: str | None = None

    @field_validator("extra_columns", "drop_columns", mode="before")
    @classmethod
    def validate_columns(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("Export columns must be provided as a list of strings.")
        normalized: list[str] = []
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError("Export columns must be strings.")
            trimmed = entry.strip()
            if not trimmed:
                raise ValueError("Export columns must be non-empty strings.")
            normalized.append(trimmed)
        return normalized

    @field_validator("answer_column", mode="before")
    @classmethod
    def validate_answer_column(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("answer_column must be a string.")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("answer_column must be a non-empty string.")
        return trimmed


class EnvironmentConfigSchema(BaseModel):
    """Schema for environment configuration entries (keyed by identifier)."""

    id: str | None = Field(None, description="Optional environment identifier (legacy list format).")
    module: str | None = Field(None, description="Optional module override when the ID differs from the import path.")
    num_examples: int = Field(5, description="Number of examples to evaluate (-1 for all).")
    rollouts_per_example: int = Field(1, description="Number of rollouts to perform per example.")
    max_concurrent: int | None = Field(
        None, description="Maximum number of concurrent requests when running the environment."
    )
    interleave_scoring: bool = Field(True, description="Whether to interleave scoring requests.")
    state_columns: list[str] | None = Field(
        default=None, description="Optional state columns to persist in job outputs."
    )
    save_every: int | None = Field(default=None, description="Persist intermediate results every N examples when set.")
    print_results: bool = Field(False, description="Print environment results to stdout.")
    verbose: bool | None = Field(None, description="Override per-environment verbosity.")
    env_args: dict[str, Any] = Field(default_factory=dict)
    rerun: bool = Field(
        False,
        description="Re-run jobs for this environment when resuming/regenerating even if previously completed.",
    )
    matrix: dict[str, list[Any]] | None = Field(default=None, description="Parameter sweeps for expansion.")
    matrix_exclude: list[dict[str, Any]] | None = Field(default=None, description="List of matrix patterns to exclude.")
    matrix_id_format: str | None = Field(default=None, description="Optional format string for matrix variant IDs.")
    matrix_base_id: str | None = Field(default=None, exclude=True)
    export: EnvironmentExportConfig | None = Field(
        default=None,
        description="Optional export customization (keep/drop columns, prompt settings).",
    )

    @field_validator("num_examples")
    @classmethod
    def validate_num_examples(cls, value: int) -> int:
        if value == -1 or value >= 1:
            return value
        msg = "num_examples must be -1 (all) or >= 1."
        raise ValueError(msg)

    @field_validator("env_args")
    @classmethod
    def default_env_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @field_validator("rollouts_per_example")
    @classmethod
    def validate_rollouts_per_example(cls, value: int) -> int:
        if value >= 1:
            return value
        raise ValueError("rollouts_per_example must be >= 1.")

    @field_validator("max_concurrent")
    @classmethod
    def validate_max_concurrent(cls, value: int | None) -> int | None:
        if value is None or value >= 1:
            return value
        raise ValueError("max_concurrent must be >= 1 when provided.")

    @field_validator("state_columns")
    @classmethod
    def validate_state_columns(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("state_columns must be a list of strings when provided.")
        return [str(item) for item in value]

    @field_validator("save_every")
    @classmethod
    def validate_save_every(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value >= 1:
            return value
        raise ValueError("save_every must be >= 1 when provided.")

    @field_validator("matrix", mode="before")
    @classmethod
    def validate_matrix(cls, value: Any) -> dict[str, list[Any]] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("matrix must be a mapping of parameter names to value lists.")
        normalized: dict[str, list[Any]] = {}
        for key, items in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("matrix keys must be non-empty strings.")
            if isinstance(items, tuple):
                items = list(items)
            elif not isinstance(items, list):
                raise ValueError(f"matrix['{key}'] must be a list of values.")
            if not items:
                raise ValueError(f"matrix['{key}'] must contain at least one value.")
            normalized[key] = list(items)
        return normalized

    @field_validator("matrix_exclude", mode="before")
    @classmethod
    def validate_matrix_exclude(cls, value: Any) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("matrix_exclude must be a list of mappings.")
        normalized: list[dict[str, Any]] = []
        for entry in value:
            if not isinstance(entry, dict):
                raise ValueError("matrix_exclude entries must be mappings.")
            normalized.append(dict(entry))
        return normalized

    @field_validator("matrix_id_format")
    @classmethod
    def validate_matrix_id_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError("matrix_id_format must be a non-empty string when provided.")
        return value

    @model_validator(mode="after")
    def validate_matrix_constraints(self) -> "EnvironmentConfigSchema":
        matrix = self.matrix or {}
        if matrix:
            base_id = self.id or "<environment>"
            for key in matrix:
                if key in RESERVED_MATRIX_KEYS:
                    raise ValueError(f"environment '{base_id}' matrix cannot vary '{key}'.")
            matrix_keys = set(matrix)
            if self.matrix_exclude:
                for pattern in self.matrix_exclude:
                    invalid_keys = set(pattern) - matrix_keys
                    if invalid_keys:
                        invalid = ", ".join(sorted(invalid_keys))
                        raise ValueError(
                            f"environment '{base_id}' matrix_exclude entry references unknown keys: {invalid}."
                        )
        return self


class JobConfigSchema(BaseModel):
    """Schema for job entries mapping models to environments."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | dict[str, Any] = Field(..., description="Reference to a defined model id or inline model definition.")
    env: str | list[str] = Field(..., description="Reference to an environment id or list of ids.")
    env_args: dict[str, Any] = Field(default_factory=dict)
    sampling_args: dict[str, Any] = Field(default_factory=dict)
    name: str | None = Field(default=None, description="Optional human-friendly job label.")
    sleep: float | None = Field(default=None, ge=0, description="Optional delay (in seconds) after this job.")


DEFAULT_RUN_OUTPUT_DIR = Path("runs") / "raw"


class RunConfigSchema(BaseModel):
    """Top-level configuration for unified CLI runs."""

    name: str = Field("benchmark", description="Human readable run name.")
    models: dict[str, ModelConfigSchema] = Field(default_factory=dict, description="Map of model id -> configuration.")
    envs: dict[str, EnvironmentConfigSchema] = Field(
        ..., description="Map of environment id -> configuration.", min_length=1
    )
    jobs: list[JobConfigSchema] = Field(default_factory=list)
    output_dir: Path = Field(default_factory=lambda: DEFAULT_RUN_OUTPUT_DIR)


__all__ = [
    "ModelConfigSchema",
    "EnvironmentConfigSchema",
    "EnvironmentExportConfig",
    "JobConfigSchema",
    "RunConfigSchema",
    "RESERVED_MATRIX_KEYS",
]
