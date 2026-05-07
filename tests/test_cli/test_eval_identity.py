from __future__ import annotations

from pathlib import Path

import pytest

from medarc_verifiers.cli.eval_identity import (
    MEDARC_CONFIG_FINGERPRINT_KEY,
    MEDARC_CONFIG_FINGERPRINT_PAYLOAD_KEY,
    MEDARC_VARIANT_ID_KEY,
    MEDARC_VARIANT_PAYLOAD_KEY,
    build_fingerprint_payload,
    config_fingerprint,
    generate_variant_id,
    metadata_identity_fields,
    normalize_semantic_sampling_args,
    plan_eval_paths,
    slug_component,
)


def test_unique_model_env_path_uses_plain_dataset_directory(tmp_path: Path) -> None:
    [plan] = plan_eval_paths(
        [{"model": "openai/gpt-5-mini", "env_id": "medqa"}],
        output_root=tmp_path / "runs" / "evals",
    )

    assert plan.identity.model_id == "openai/gpt-5-mini"
    assert plan.identity.env_id == "medqa"
    assert plan.identity.variant_id is None
    assert plan.identity.variant_payload is None
    assert plan.results_path == tmp_path / "runs" / "evals" / "openai-gpt-5-mini" / "medqa"


def test_duplicate_model_env_paths_use_deterministic_variants(tmp_path: Path) -> None:
    plans = plan_eval_paths(
        [
            {"model": "gpt-5-mini", "env_id": "medqa", "env_args": {"shuffle_seed": 1618}},
            {"model": "gpt-5-mini", "env_id": "medqa", "env_args": {"shuffle_seed": 9331}},
        ],
        output_root=tmp_path,
    )

    assert [plan.identity.variant_id for plan in plans] == ["env_args.shuffle_seed-1618", "env_args.shuffle_seed-9331"]
    assert plans[0].identity.variant_payload == {"env_args": {"shuffle_seed": 1618}}
    assert plans[1].identity.variant_payload == {"env_args": {"shuffle_seed": 9331}}
    assert plans[0].results_path == tmp_path / "gpt-5-mini" / "medqa" / "env_args.shuffle_seed-1618"
    assert plans[1].results_path == tmp_path / "gpt-5-mini" / "medqa" / "env_args.shuffle_seed-9331"


def test_duplicate_model_env_baseline_gets_explicit_variant(tmp_path: Path) -> None:
    plans = plan_eval_paths(
        [
            {"model": "gpt-5-mini", "env_id": "medqa"},
            {"model": "gpt-5-mini", "env_id": "medqa", "env_args": {"shuffle_seed": 1618}},
        ],
        output_root=tmp_path,
    )

    assert [plan.identity.variant_id for plan in plans] == ["baseline", "env_args.shuffle_seed-1618"]
    assert plans[0].identity.variant_payload == {"env_args": {}}
    assert plans[0].results_path == tmp_path / "gpt-5-mini" / "medqa" / "baseline"


def test_duplicate_model_env_variant_can_use_sampling_args(tmp_path: Path) -> None:
    plans = plan_eval_paths(
        [
            {"model": "gpt-5-mini", "env_id": "medqa", "sampling_args": {"temperature": 0.0}},
            {"model": "gpt-5-mini", "env_id": "medqa", "sampling_args": {"temperature": 0.7}},
        ],
        output_root=tmp_path,
    )

    assert [plan.identity.variant_id for plan in plans] == [
        "sampling_args.temperature-0.0",
        "sampling_args.temperature-0.7",
    ]


def test_duplicate_model_env_variant_uses_only_differing_nested_keys(tmp_path: Path) -> None:
    common_env_args = {
        "judge_model": ["openai/gpt-5-mini", "google/gemini-3-flash-preview"],
        "judge_base_url": "https://api.pinference.ai/api/v1",
    }

    plans = plan_eval_paths(
        [
            {"model": "gpt-5-mini", "env_id": "medrbench", "env_args": {**common_env_args, "task": "oracle"}},
            {"model": "gpt-5-mini", "env_id": "medrbench", "env_args": {**common_env_args, "task": "1turn"}},
            {"model": "gpt-5-mini", "env_id": "medrbench", "env_args": {**common_env_args, "task": "free_turn"}},
        ],
        output_root=tmp_path,
    )

    assert [plan.identity.variant_id for plan in plans] == [
        "env_args.task-oracle",
        "env_args.task-1turn",
        "env_args.task-free_turn",
    ]
    assert plans[0].identity.variant_payload == {"env_args": {"task": "oracle"}}


def test_duplicate_model_env_variant_canonicalizes_sampling_args(tmp_path: Path) -> None:
    plans = plan_eval_paths(
        [
            {"model": "gpt-5-mini", "env_id": "medqa", "sampling_args": {"reasoning_effort": "medium"}},
            {
                "model": "gpt-5-mini",
                "env_id": "medqa",
                "sampling_args": {"extra_body": {"reasoning": {"effort": "high"}}},
            },
        ],
        output_root=tmp_path,
    )

    assert [plan.identity.variant_id for plan in plans] == [
        "sampling_args.reasoning_effort-medium",
        "sampling_args.reasoning_effort-high",
    ]


def test_long_nested_variant_values_use_stable_fingerprint() -> None:
    payload = {
        "env_args": {
            "rubric": {
                "criteria": ["clinically grounded", "concise", "safe"],
                "description": "x" * 240,
            }
        }
    }

    variant_id = generate_variant_id(payload)

    assert len(variant_id) <= 160
    assert variant_id.endswith(generate_variant_id(payload)[-12:])
    assert "env_args.rubric-hash" in variant_id


def test_fingerprint_stable_across_key_ordering() -> None:
    left = {
        "env_id": "medqa",
        "model": "gpt-5-mini",
        "env_args": {"b": 2, "a": 1},
        "sampling_args": {"top_p": 0.9, "temperature": 0.1},
        "num_examples": 10,
        "rollouts_per_example": 1,
    }
    right = {
        "rollouts_per_example": 1,
        "num_examples": 10,
        "sampling_args": {"temperature": 0.1, "top_p": 0.9},
        "env_args": {"a": 1, "b": 2},
        "model": "gpt-5-mini",
        "env_id": "medqa",
    }

    assert config_fingerprint(left) == config_fingerprint(right)
    assert build_fingerprint_payload(left) == build_fingerprint_payload(right)


@pytest.mark.parametrize(
    "changed",
    [
        {"env_args": {"shuffle_seed": 9331}},
        {"sampling_args": {"temperature": 0.8}},
        {"max_tokens": 1024},
        {"num_examples": 11},
        {"rollouts_per_example": 2},
    ],
)
def test_fingerprint_changes_for_semantic_benchmark_changes(changed: dict[str, object]) -> None:
    base = {
        "env_id": "medqa",
        "model": "gpt-5-mini",
        "env_args": {"shuffle_seed": 1618},
        "sampling_args": {"temperature": 0.2},
        "num_examples": 10,
        "rollouts_per_example": 1,
    }
    candidate = {**base, **changed}

    assert config_fingerprint(base) != config_fingerprint(candidate)


@pytest.mark.parametrize(
    "changed",
    [
        {"provider": "openai"},
        {"api_base_url": "http://localhost:9000/v1"},
        {"endpoint_id": "local-alias"},
        {"api_key_var": "LOCAL_KEY"},
        {"api_client_type": "openai_chat_completions"},
        {"timeout": 120},
        {"max_concurrent": 1},
        {"max_retries": 5},
        {"headers": {"X-Prime-Team-ID": "team"}},
        {"sampling_args": {"temperature": 0.2, "extra_body": {"usage": {"include": True}}}},
    ],
)
def test_fingerprint_ignores_provider_transport_and_runtime_changes(changed: dict[str, object]) -> None:
    base = {
        "env_id": "medqa",
        "model": "gpt-5-mini",
        "env_args": {"shuffle_seed": 1618},
        "sampling_args": {"temperature": 0.2},
        "num_examples": 10,
        "rollouts_per_example": 1,
    }
    candidate = {**base, **changed}

    assert config_fingerprint(base) == config_fingerprint(candidate)


def test_variant_planning_ignores_runtime_fields_in_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Deterministic eval path collision"):
        plan_eval_paths(
            [
                {"model": "gpt-5-mini", "env_id": "medqa", "max_concurrent": 1},
                {"model": "gpt-5-mini", "env_id": "medqa", "max_concurrent": 32, "timeout": 120},
            ],
            output_root=tmp_path,
        )


def test_reasoning_effort_shapes_fingerprint_identically() -> None:
    native = {
        "env_id": "medqa",
        "model": "gpt-5-mini",
        "sampling_args": {"reasoning_effort": "medium", "temperature": 0.2},
        "num_examples": 10,
        "rollouts_per_example": 1,
    }
    openrouter = {
        "env_id": "medqa",
        "model": "gpt-5-mini",
        "sampling_args": {"extra_body": {"reasoning": {"effort": "medium"}}, "temperature": 0.2},
        "num_examples": 10,
        "rollouts_per_example": 1,
    }

    assert config_fingerprint(native) == config_fingerprint(openrouter)
    assert build_fingerprint_payload(native)["sampling_args"] == {
        "reasoning_effort": "medium",
        "temperature": 0.2,
    }


def test_top_level_sampling_aliases_match_sampling_args_shape() -> None:
    top_level = {
        "env_id": "medqa",
        "model": "gpt-5-mini",
        "temperature": 0.2,
        "max_tokens": 256,
        "num_examples": 10,
        "rollouts_per_example": 1,
    }
    nested = {
        "env_id": "medqa",
        "model": "gpt-5-mini",
        "sampling_args": {"temperature": 0.2, "max_tokens": 256},
        "num_examples": 10,
        "rollouts_per_example": 1,
    }

    assert config_fingerprint(top_level) == config_fingerprint(nested)
    assert build_fingerprint_payload(top_level)["sampling_args"] == {"max_tokens": 256, "temperature": 0.2}


def test_extra_body_semantic_args_match_top_level_shape() -> None:
    assert normalize_semantic_sampling_args({"top_k": 20}) == normalize_semantic_sampling_args(
        {"extra_body": {"top_k": 20}}
    )


def test_unknown_sampling_args_pass_through_fingerprint() -> None:
    assert normalize_semantic_sampling_args({"vendor_knob": True}) == {"vendor_knob": True}
    assert normalize_semantic_sampling_args({"extra_body": {"vendor_knob": True}}) == {"vendor_knob": True}


def test_endpoint_alias_without_resolved_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="resolved 'model'"):
        config_fingerprint({"endpoint_id": "gpt-alias", "env_id": "medqa"})


def test_metadata_identity_fields_include_planned_keys(tmp_path: Path) -> None:
    plan = plan_eval_paths(
        [
            {"model": "gpt-5-mini", "env_id": "medqa", "env_args": {"shuffle_seed": 1618}},
            {"model": "gpt-5-mini", "env_id": "medqa", "env_args": {"shuffle_seed": 9331}},
        ],
        output_root=tmp_path,
    )[0]
    config = {
        "model": "gpt-5-mini",
        "env_id": "medqa",
        "env_args": {"shuffle_seed": 1618},
        "sampling_args": {"temperature": 0.2},
        "num_examples": 10,
        "rollouts_per_example": 1,
    }

    fields = metadata_identity_fields(config, plan.identity)

    assert fields[MEDARC_CONFIG_FINGERPRINT_KEY] == config_fingerprint(config)
    assert fields[MEDARC_CONFIG_FINGERPRINT_PAYLOAD_KEY] == build_fingerprint_payload(config)
    assert fields[MEDARC_VARIANT_ID_KEY] == "env_args.shuffle_seed-1618"
    assert fields[MEDARC_VARIANT_PAYLOAD_KEY] == {"env_args": {"shuffle_seed": 1618}}


def test_slug_component_is_path_safe_and_stable_for_long_values() -> None:
    slug = slug_component(" openai/gpt-5:mini " + "x" * 120)

    assert "/" not in slug
    assert ":" not in slug
    assert len(slug) <= 80
    assert slug == slug_component(" openai/gpt-5:mini " + "x" * 120)
