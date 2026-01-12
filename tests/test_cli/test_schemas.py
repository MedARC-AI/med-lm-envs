from __future__ import annotations

import pytest

from medarc_verifiers.cli._schemas import (
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
            "extra_columns": ["answer", " score "],
            "drop_columns": ["raw_state"],
            "combine_rollouts": False,
            "answer_column": "ground_truth",
        },
    )
    assert env.export is not None
    assert env.export.extra_columns == ["answer", "score"]
    assert env.export.drop_columns == ["raw_state"]
    assert env.export.combine_rollouts is False
    assert env.export.answer_column == "ground_truth"


def test_environment_export_config_invalid_column_type_raises() -> None:
    with pytest.raises(ValueError, match="Export columns must be provided as a list of strings."):
        EnvironmentExportConfig(extra_columns=123)
