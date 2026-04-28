"""Small schemas still shared by process export configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EnvironmentExportConfig(BaseModel):
    """Optional export customization embedded in legacy environment configs."""

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
    """Legacy environment YAML entry schema used for process export overrides."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    module: str | None = None
    env_args: dict[str, Any] = Field(default_factory=dict)
    matrix_base_id: str | None = Field(default=None, exclude=True)
    export: EnvironmentExportConfig | None = None

    @field_validator("env_args")
    @classmethod
    def default_env_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)


__all__ = [
    "EnvironmentConfigSchema",
    "EnvironmentExportConfig",
]
