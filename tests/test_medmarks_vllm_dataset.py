from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient

from benchmarks.medmarks_vllm.bench_client import run_requests
from benchmarks.medmarks_vllm import generate_dataset as gen


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "include_audit_metadata": False,
        "seed": 123,
        "tokenizer": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _export(item: dict, env: object) -> list[dict]:
    converter = OpenAIChatCompletionsClient(object())
    with asyncio.Runner() as runner:
        return gen.export_openai_messages(item, env=env, converter=converter, runner=runner)


def test_load_variants_keeps_current_medmarks_vllm_subset_and_tool_variant() -> None:
    variants = gen.load_variants(Path("configs/medmarks-verified.toml"))

    assert {variant.env_id for variant in variants} == gen.TARGET_ENVS
    assert any(variant.env_id == "medcalc_bench" and variant.variant_id == "tools" for variant in variants)
    assert len({gen.variant_key(variant) for variant in variants}) == len(variants)


def test_string_prompt_exports_single_user_message() -> None:
    messages = _export({"prompt": "hello"}, SimpleNamespace(system_prompt=None))

    assert messages == [{"role": "user", "content": "hello"}]


def test_chat_prompt_preserves_system_and_user_roles() -> None:
    messages = _export(
        {
            "prompt": [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "question text"},
            ]
        },
        SimpleNamespace(system_prompt="ignored because prompt is already structured"),
    )

    assert messages == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "question text"},
    ]


def test_flattened_system_user_string_is_rejected() -> None:
    with pytest.raises(gen.UnsupportedRowError, match="flattened System/User"):
        _export({"prompt": "System:\nBe precise.\n\nUser:\nWhat now?"}, SimpleNamespace(system_prompt=None))


def test_string_prompt_with_authoritative_question_and_system_is_reconstructed() -> None:
    messages = _export(
        {"prompt": "What now?", "question": "What now?"},
        SimpleNamespace(system_prompt="Be precise.", few_shot=None),
    )

    assert messages == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "What now?"},
    ]


def test_render_variant_uses_get_eval_dataset_for_lazy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    class LazyEnv:
        system_prompt = None
        tool_defs = None

        @property
        def eval_dataset(self) -> Dataset:  # pragma: no cover - would fail the acceptance contract
            raise AssertionError("render_variant must use get_eval_dataset(), not eval_dataset")

        def get_eval_dataset(self) -> Dataset:
            return Dataset.from_list([{"prompt": [{"role": "user", "content": "hello"}], "question": "hello"}])

    monkeypatch.setattr(gen, "resolve_load_environment", lambda env_id, env_dir: lambda **env_args: LazyEnv())
    rows = gen.render_variant(
        gen.Variant("careqa", "base", "base", {}, "eval"),
        Path("environments"),
        _args(),
        tokenizer=None,
    )

    assert rows[0].record["messages"] == [{"role": "user", "content": "hello"}]
    assert "prompt" not in rows[0].record
    assert "output_tokens" not in rows[0].record


def test_tool_variant_is_skipped_by_default() -> None:
    variant = gen.Variant(
        "medcalc_bench",
        "tools",
        "tools",
        {"add_python_tool": True, "add_calculator_tool": True},
        "eval",
    )

    assert gen.is_tool_enabled_variant(variant)
    assert "tool-enabled" in gen.skipped_variant(variant, "tool-enabled variants are excluded")["error"]


def test_stats_definitions_are_for_request_messages_not_output_lengths() -> None:
    row = gen.RenderedRow(
        record={
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}],
            "env_id": "careqa",
            "variant_id": "base",
        },
        prompt_chars=6,
        input_tokens_approx=4,
        input_tokens=None,
    )

    stats = gen.build_stats(
        selected_rows=[row],
        rows_by_variant={"careqa::base": [row]},
        failures=[],
        args=_args(),
        target_size=1,
        tokenizer_exact=False,
    )

    assert stats["artifact_schema"] == "openai-chat-requests-v1"
    assert "output_tokens" not in stats
    assert "fixed_output_tokens" not in stats
    assert stats["prompt_chars_definition"].startswith("Sum of text content lengths")
    assert stats["input_tokens_definition"].startswith("Exact tokenizer count")
    assert stats["input_tokens"]["min"] is None
    assert stats["input_tokens_approx"]["min"] == 4
    assert stats["token_buckets_source"] == "input_tokens_approx"


def test_adapter_rejects_legacy_prompt_rows() -> None:
    with pytest.raises(ValueError, match="prompt/output_tokens"):
        run_requests.validate_request_row({"prompt": "hello", "messages": [{"role": "user", "content": "hello"}]}, 1)


def test_adapter_rejects_reserved_sampling_request_keys() -> None:
    with pytest.raises(ValueError, match="reserved request keys"):
        run_requests.validate_request_row(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "sampling_args": {"max_tokens": 99},
            },
            1,
        )
