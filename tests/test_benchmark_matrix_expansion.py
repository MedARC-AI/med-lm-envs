from __future__ import annotations

from pathlib import Path

import pytest

from medarc_verifiers.cli._benchmark_utils import build_run_config, expand_jobs


def _base_config(tmp_path: Path) -> dict:
    return {
        "name": "matrix",
        "output_dir": str(tmp_path / "runs"),
        "models": [{"id": "model-a"}],
    }


def test_matrix_expansion_generates_variants(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["envs"] = [
        {
            "id": "env-base",
            "module": "env_module",
            "num_examples": 5,
            "rollouts_per_example": 1,
            "env_args": {"shuffle_answers": True},
            "matrix": {
                "difficulty": ["easy", "hard"],
                "shuffle_seed": [1618, 9331],
            },
            "matrix_id_format": "{base}-{difficulty}-s{shuffle_seed}",
        }
    ]

    run_config = build_run_config(config, tmp_path)
    env_ids = sorted(run_config.envs)
    assert env_ids == [
        "env-base-easy-s1618",
        "env-base-easy-s9331",
        "env-base-hard-s1618",
        "env-base-hard-s9331",
    ]
    sample_env = run_config.envs["env-base-easy-s1618"]
    assert sample_env.rollouts_per_example == 1
    assert sample_env.env_args["shuffle_answers"] is True
    assert sample_env.env_args["difficulty"] == "easy"
    assert sample_env.env_args["shuffle_seed"] == 1618


def test_matrix_exclude_and_null_values(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["envs"] = [
        {
            "id": "env-difficulty",
            "module": "env_module",
            "env_args": {"shuffle_answers": False},
            "matrix": {
                "difficulty": ["easy", "medium"],
                "shuffle_seed": [None, 5],
            },
            "matrix_exclude": [{"difficulty": "easy", "shuffle_seed": 5}],
        }
    ]

    run_config = build_run_config(config, tmp_path)
    env_ids = sorted(run_config.envs)
    assert env_ids == [
        "env-difficulty-difficulty-easy",
        "env-difficulty-difficulty-medium",
        "env-difficulty-difficulty-medium-shuffle_seed-5",
    ]
    base_variant = run_config.envs["env-difficulty-difficulty-easy"]
    assert base_variant.env_args["difficulty"] == "easy"
    assert "shuffle_seed" not in base_variant.env_args
    seeded_variant = run_config.envs["env-difficulty-difficulty-medium-shuffle_seed-5"]
    assert seeded_variant.env_args["difficulty"] == "medium"
    assert seeded_variant.env_args["shuffle_seed"] == 5


def test_matrix_duplicate_ids_raise(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["envs"] = [
        {
            "id": "env-dup",
            "module": "env_module",
            "matrix": {
                "shuffle_seed": [101, 101],
            },
        }
    ]

    with pytest.raises(ValueError, match="duplicate id"):
        build_run_config(config, tmp_path)


def test_matrix_exclude_unknown_key_raises(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["envs"] = [
        {
            "id": "env-error",
            "module": "env_module",
            "matrix": {
                "difficulty": ["easy"],
            },
            "matrix_exclude": [{"shuffle_seed": 1}],
        }
    ]

    with pytest.raises(ValueError, match="matrix_exclude entry references unknown keys"):
        build_run_config(config, tmp_path)


def test_matrix_updates_scalar_fields(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["envs"] = [
        {
            "id": "env-top",
            "module": "env_module",
            "num_examples": 3,
            "matrix": {
                "num_examples": [5, 7],
                "rollouts_per_example": [1, 2],
            },
            "matrix_id_format": "{base}-n{num_examples}-r{rollouts_per_example}",
        }
    ]

    run_config = build_run_config(config, tmp_path)
    assert sorted(run_config.envs) == [
        "env-top-n5-r1",
        "env-top-n5-r2",
        "env-top-n7-r1",
        "env-top-n7-r2",
    ]
    assert run_config.envs["env-top-n5-r1"].num_examples == 5
    assert run_config.envs["env-top-n5-r1"].rollouts_per_example == 1
    assert run_config.envs["env-top-n7-r2"].num_examples == 7
    assert run_config.envs["env-top-n7-r2"].rollouts_per_example == 2


def test_inline_model_definition_in_job(tmp_path: Path) -> None:
    config = {
        "name": "inline-model",
        "output_dir": str(tmp_path / "runs"),
        "envs": [
            {
                "id": "env-inline",
                "module": "env_module",
            }
        ],
        "jobs": [
            {
                "model": {
                    "id": "inline-model-id",
                    "params": {"model": "openai/inline-test", "max_tokens": 256},
                },
                "env": "env-inline",
            }
        ],
    }

    run_config = build_run_config(config, tmp_path)
    assert "inline-model-id" in run_config.models

    resolved_jobs = expand_jobs(run_config)
    assert len(resolved_jobs) == 1
    resolved_job = resolved_jobs[0]
    assert resolved_job.model.id == "inline-model-id"
    assert resolved_job.model.params.model == "openai/inline-test"


def test_top_level_job_model_params(tmp_path: Path) -> None:
    config = {
        "name": "top-level",
        "output_dir": str(tmp_path / "runs"),
        "envs": [
            {
                "id": "env-top-level",
                "module": "env_module",
            }
        ],
        "jobs": [
            {
                "model": "openai/gpt-oss-20b",
                "api_base_url": "http://localhost:8000/v1",
                "sampling_args": {"temperature": 0.3},
                "env_args": {"max_tokens": 1024},
                "envs": ["env-top-level"],
            }
        ],
    }

    run_config = build_run_config(config, tmp_path)
    assert "openai/gpt-oss-20b" in run_config.models
    model_cfg = run_config.models["openai/gpt-oss-20b"]
    assert model_cfg.params.api_base_url == "http://localhost:8000/v1"
    assert model_cfg.params.model == "openai/gpt-oss-20b"
    assert model_cfg.params.sampling_args == {"temperature": 0.3}
    assert model_cfg.params.env_args == {"max_tokens": 1024}

    resolved_jobs = expand_jobs(run_config)
    assert resolved_jobs[0].name == "openai/gpt-oss-20b"


def test_model_id_override_defaults_name(tmp_path: Path) -> None:
    config = {
        "name": "model-id",
        "output_dir": str(tmp_path / "runs"),
        "envs": [
            {
                "id": "env-id",
                "module": "env_module",
            }
        ],
        "jobs": [
            {
                "model": "openai/gpt-oss-20b",
                "model_id": "gpt-oss-20b",
                "envs": ["env-id"],
            }
        ],
    }

    run_config = build_run_config(config, tmp_path)
    assert "gpt-oss-20b" in run_config.models
    model_cfg = run_config.models["gpt-oss-20b"]
    assert model_cfg.params.model == "openai/gpt-oss-20b"
    assert run_config.jobs[0].name == "openai/gpt-oss-20b"
