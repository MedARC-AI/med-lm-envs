from __future__ import annotations

from pathlib import Path

import pytest

from medarc_verifiers.cli.eval_identity import (
    BASE_VARIANT_ID,
    generate_variant_id,
    plan_eval_paths,
    slug_component,
)


def test_unique_model_env_path_uses_base_variant_directory(tmp_path: Path) -> None:
    [plan] = plan_eval_paths(
        [{"model": "openai/gpt-5-mini", "env_id": "medqa"}],
        output_root=tmp_path / "runs" / "evals",
    )

    assert plan.identity.model_id == "openai/gpt-5-mini"
    assert plan.identity.env_id == "medqa"
    assert plan.identity.variant_id == BASE_VARIANT_ID
    assert plan.results_path == tmp_path / "runs" / "evals" / "openai-gpt-5-mini" / "medqa" / "base"


def test_explicit_variant_id_controls_variant_directory(tmp_path: Path) -> None:
    [plan] = plan_eval_paths(
        [{"model": "gpt-5-mini", "env_id": "medqa", "variant_id": "shuffle_seed-1618"}],
        output_root=tmp_path,
    )

    assert plan.identity.variant_id == "shuffle_seed-1618"
    assert plan.results_path == tmp_path / "gpt-5-mini" / "medqa" / "shuffle_seed-1618"


def test_name_is_variant_id_alias(tmp_path: Path) -> None:
    [plan] = plan_eval_paths(
        [{"model": "gpt-5-mini", "env_id": "medqa", "name": "seed-1618"}],
        output_root=tmp_path,
    )

    assert plan.identity.variant_id == "seed-1618"
    assert plan.results_path == tmp_path / "gpt-5-mini" / "medqa" / "seed-1618"


def test_name_can_template_expanded_env_args(tmp_path: Path) -> None:
    [plan] = plan_eval_paths(
        [
            {
                "model": "gpt-5-mini",
                "env_id": "medqa",
                "env_args": {"shuffle_seed": 1618},
                "name": "shuffle_seed-{env_args.shuffle_seed}",
            }
        ],
        output_root=tmp_path,
    )

    assert plan.identity.variant_id == "shuffle_seed-1618"


def test_matching_name_and_variant_id_are_allowed(tmp_path: Path) -> None:
    [plan] = plan_eval_paths(
        [{"model": "gpt-5-mini", "env_id": "medqa", "name": "base", "variant_id": "base"}],
        output_root=tmp_path,
    )

    assert plan.identity.variant_id == "base"


def test_conflicting_name_and_variant_id_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conflicting variant_id/name"):
        plan_eval_paths(
            [{"model": "gpt-5-mini", "env_id": "medqa", "name": "left", "variant_id": "right"}],
            output_root=tmp_path,
        )


def test_duplicate_model_env_requires_explicit_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate TOML eval identity"):
        plan_eval_paths(
            [
                {"model": "gpt-5-mini", "env_id": "medqa"},
                {"model": "gpt-5-mini", "env_id": "medqa"},
            ],
            output_root=tmp_path,
        )


def test_same_variant_condition_across_models_keeps_same_variant_id(tmp_path: Path) -> None:
    plans = plan_eval_paths(
        [
            {"model": "gpt-5-mini", "env_id": "medqa", "variant_id": "seed-1618"},
            {"model": "gpt-5", "env_id": "medqa", "variant_id": "seed-1618"},
        ],
        output_root=tmp_path,
    )

    assert [plan.identity.variant_id for plan in plans] == ["seed-1618", "seed-1618"]
    assert plans[0].results_path == tmp_path / "gpt-5-mini" / "medqa" / "seed-1618"
    assert plans[1].results_path == tmp_path / "gpt-5" / "medqa" / "seed-1618"


def test_slug_collisions_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model slug collision"):
        plan_eval_paths(
            [
                {"model": "openai/gpt", "env_id": "medqa"},
                {"model": "openai:gpt", "env_id": "pubmedqa"},
            ],
            output_root=tmp_path,
        )


def test_legacy_variant_generator_remains_stable_for_export_config_lookup() -> None:
    payload = {"env_args": {"shuffle_seed": 1618}}

    assert generate_variant_id(payload) == "env_args.shuffle_seed-1618"


def test_slug_component_is_path_safe_and_stable_for_long_values() -> None:
    slug = slug_component(" openai/gpt-5:mini " + "x" * 120)

    assert "/" not in slug
    assert ":" not in slug
    assert len(slug) <= 80
    assert slug == slug_component(" openai/gpt-5:mini " + "x" * 120)
