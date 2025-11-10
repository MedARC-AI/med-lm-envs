from __future__ import annotations

import pytest

from medarc_verifiers.cli_new._schemas import (
    EnvironmentConfigSchema,
    EnvironmentExportConfig,
    ModelConfigSchema,
)


def test_model_params_merge_matches_explicit_definition() -> None:
    explicit = ModelConfigSchema(
        id="demo",
        model="gpt-mini",
        env_args={"split": "dev"},
        env_overrides={"medqa": {"temperature": 0.2}},
    )
    legacy = ModelConfigSchema(
        id="demo",
        params={
            "model": "gpt-mini",
            "env_args": {"split": "dev"},
            "env_overrides": {"medqa": {"temperature": 0.2}},
        },
    )

    assert legacy.model_dump() == explicit.model_dump()


def test_environment_matrix_exclude_with_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="matrix_exclude entry references unknown keys"):
        EnvironmentConfigSchema(
            id="medqa",
            matrix={"num_examples": [5]},
            matrix_exclude=[{"unknown_key": 1}],
        )


def test_environment_export_config_validates_columns() -> None:
    env = EnvironmentConfigSchema(
        id="medqa",
        module="environments.medqa",
        export={
            "keep_columns": ["answer", " score "],
            "drop_columns": ["raw_state"],
            "include_prompt_completion": True,
            "combine_rollouts": False,
        },
    )
    assert env.export is not None
    assert env.export.keep_columns == ["answer", "score"]
    assert env.export.drop_columns == ["raw_state"]
    assert env.export.include_prompt_completion is True
    assert env.export.combine_rollouts is False


def test_environment_export_config_invalid_column_type_raises() -> None:
    with pytest.raises(ValueError, match="Export columns must be provided as a list of strings."):
        EnvironmentExportConfig(keep_columns=123)
