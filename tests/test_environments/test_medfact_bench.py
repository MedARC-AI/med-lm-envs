from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

import pytest
from datasets import Dataset
from verifiers.types import ClientConfig, Response, ResponseMessage

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_ROOT = REPO_ROOT / "environments" / "medfact_bench"
if str(ENVIRONMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_ROOT))

from medfact_bench import (  # noqa: E402
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_DATASET_COUNTS,
    INVALID_LABEL,
    MedFactScoreParser,
    MedFactSubset,
    load_environment,
    parseable_score,
    prediction_label,
    strict_format,
)
import medfact_bench.environment as medfact_environment  # noqa: E402

REAL_DATASET_PATH = Path(
    "/Users/kouatemuhamed/Claude/Projects/MedARC Agentic Medical Fact Verifier/"
    "datasets/MedFact-Bench/MedFact-Bench.parquet"
)


def _source_row(
    dataset: str,
    label: str,
    *,
    claim: str | None = None,
    source: str | None = None,
    system_prompt: str = "Canonical system prompt",
) -> dict[str, str]:
    claim = claim or f"Claim for {dataset}"
    source = source or f"Source for {dataset}"
    return {
        "dataset": dataset,
        "claim": claim,
        "source": source,
        "label": label,
        "system_prompt": system_prompt,
        "user_prompt": f"Article:\n{source}\n\nClaim:\n{claim}",
    }


def _source_rows() -> list[dict[str, str]]:
    return [
        _source_row("scifact", "SUPPORT", claim="Duplicate claim", source="Duplicate source"),
        _source_row("scifact", "SUPPORT", claim="Duplicate claim", source="Duplicate source"),
        _source_row("healthver", "NEI"),
        _source_row("medaesqa", "CONTRADICT"),
        _source_row("pubmedqa-fact", "SUPPORT"),
        _source_row("bioasq-fact", "NEI"),
    ]


def _write_parquet(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "medfact-bench.parquet"
    Dataset.from_list(rows).to_parquet(path)
    return path


def _load_local_environment(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    subset: str | MedFactSubset = MedFactSubset.ALL,
):
    return load_environment(
        dataset_path=str(_write_parquet(tmp_path, rows)),
        subset=subset,
        cache_dir=str(tmp_path / "datasets-cache"),
    )


def test_loader_maps_local_parquet_and_preserves_duplicate_order(tmp_path: Path) -> None:
    rows = _source_rows()
    environment = _load_local_environment(tmp_path, rows)

    assert environment.dataset is None
    assert environment.system_prompt == "Canonical system prompt"
    assert len(environment.eval_dataset) == len(rows)
    assert set(environment.eval_dataset.column_names) == {"question", "answer", "info", "example_id", "prompt"}

    first = environment.eval_dataset[0]
    second = environment.eval_dataset[1]
    assert first["question"] == rows[0]["user_prompt"]
    assert first["answer"] == "SUPPORT"
    assert first["info"] == {"dataset": "scifact"}
    assert first["question"] == second["question"]
    assert first["answer"] == second["answer"]
    assert first["prompt"] == [
        {"role": "system", "content": "Canonical system prompt"},
        {"role": "user", "content": rows[0]["user_prompt"]},
    ]


@pytest.mark.parametrize(
    ("subset", "expected_count", "expected_dataset"),
    [
        ("scifact", 2, "scifact"),
        (MedFactSubset.HEALTHVER, 1, "healthver"),
        ("medaesqa", 1, "medaesqa"),
        ("pubmedqa-fact", 1, "pubmedqa-fact"),
        ("bioasq-fact", 1, "bioasq-fact"),
    ],
)
def test_loader_filters_subset_before_mapping(
    tmp_path: Path,
    subset: str | MedFactSubset,
    expected_count: int,
    expected_dataset: str,
) -> None:
    environment = _load_local_environment(tmp_path, _source_rows(), subset=subset)

    assert len(environment.eval_dataset) == expected_count
    assert {row["info"]["dataset"] for row in environment.eval_dataset} == {expected_dataset}


def test_loader_uses_pinned_hugging_face_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    source_dataset = Dataset.from_list(_source_rows())

    def fake_load_dataset(*args: object, **kwargs: object) -> Dataset:
        calls.append((args, kwargs))
        return source_dataset

    monkeypatch.setattr(medfact_environment, "load_dataset", fake_load_dataset)
    environment = load_environment(cache_dir="/tmp/medfact-cache")

    assert len(environment.eval_dataset) == len(source_dataset)
    assert calls == [
        (
            (DATASET_ID,),
            {
                "split": "train",
                "revision": DATASET_REVISION,
                "cache_dir": "/tmp/medfact-cache",
            },
        )
    ]


def test_loader_prefers_local_parquet_over_hugging_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_load_dataset = medfact_environment.load_dataset

    def recording_load_dataset(*args: object, **kwargs: object) -> Dataset:
        calls.append((args, kwargs))
        return original_load_dataset(*args, **kwargs)

    monkeypatch.setattr(medfact_environment, "load_dataset", recording_load_dataset)
    path = _write_parquet(tmp_path, _source_rows())
    environment = load_environment(dataset_path=str(path), cache_dir=str(tmp_path / "datasets-cache"))

    assert len(environment.eval_dataset) == 6
    assert calls[0][0] == ("parquet",)
    assert calls[0][1]["data_files"] == str(path)
    assert all(call[0] != (DATASET_ID,) for call in calls)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"dataset": "scifact"}], "missing required columns"),
        ([{**_source_rows()[0], "dataset": "unknown"}], "unsupported component values"),
        ([{**_source_rows()[0], "label": "MAYBE"}], "unsupported labels"),
        (
            [_source_rows()[0], _source_row("healthver", "NEI", system_prompt="Different system prompt")],
            "exactly one distinct system prompt",
        ),
    ],
)
def test_loader_rejects_invalid_schema(tmp_path: Path, rows: list[dict[str, str]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_environment(
            dataset_path=str(_write_parquet(tmp_path, rows)),
            cache_dir=str(tmp_path / "datasets-cache"),
        )


def test_loader_rejects_invalid_path_and_subset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_environment(dataset_path=str(tmp_path / "missing.parquet"))
    with pytest.raises(ValueError, match="Unsupported MedFact-Bench subset"):
        load_environment(dataset_path=str(_write_parquet(tmp_path, _source_rows())), subset="unknown")


@pytest.mark.parametrize(
    ("completion", "expected_score", "expected_label"),
    [
        ("<score>-2</score>", -2, "CONTRADICT"),
        ("<score> -1 </score>", -1, "CONTRADICT"),
        ("<score>0</score>", 0, "NEI"),
        ("<score>+1</score>", 1, "SUPPORT"),
        ("<score>2</score>", 2, "SUPPORT"),
        ("Before <score> 2 </score> after", 2, "SUPPORT"),
        ("<score>2</score><score>-2</score>", 2, "SUPPORT"),
        ([{"role": "assistant", "content": "<score>+1</score>"}], 1, "SUPPORT"),
    ],
)
def test_parser_accepts_valid_first_score(
    completion: str | list[dict[str, str]],
    expected_score: int,
    expected_label: str,
) -> None:
    parser = MedFactScoreParser()

    assert parser.parse_completion(completion) == expected_score
    assert prediction_label(completion, parser) == expected_label


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "<score></score>",
        "<score>1.0</score>",
        "<score>support</score>",
        "<score>3</score>",
        "<score>-3</score>",
        "<score>−1</score>",
        "<think>reasoning</think>",
        [{"role": "user", "content": "<score>1</score>"}],
    ],
)
def test_parser_rejects_invalid_scores(completion: str | list[dict[str, str]]) -> None:
    parser = MedFactScoreParser()

    assert parser.parse_completion(completion) is None
    assert prediction_label(completion, parser) == INVALID_LABEL


def test_parseability_and_strict_format_are_independent() -> None:
    parser = MedFactScoreParser()
    strict_completion = "<think>Reasoning</think><score>-1</score>"
    relaxed_completion = "Explanation first. <score>-1</score>"
    duplicate_tag_completion = "<think>Reasoning</think><score>-1</score><score>2</score>"

    assert parseable_score(strict_completion, parser) == 1.0
    assert strict_format(strict_completion, parser) == 1.0
    assert parseable_score(relaxed_completion, parser) == 1.0
    assert strict_format(relaxed_completion, parser) == 0.0
    assert parseable_score(duplicate_tag_completion, parser) == 1.0
    assert strict_format(duplicate_tag_completion, parser) == 0.0


def test_environment_scores_a_fake_model_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = _load_local_environment(tmp_path, _source_rows())

    async def fake_model_response(*_: object, **__: object) -> Response:
        return Response(
            id="fake-response",
            created=0,
            model="fake-model",
            message=ResponseMessage(
                content="<think>Evidence supports the claim.</think><score>2</score>",
                finish_reason="stop",
                is_truncated=False,
            ),
        )

    monkeypatch.setattr(environment, "get_model_response", fake_model_response)
    output = asyncio.run(
        environment.run_rollout(
            environment.eval_dataset[0],
            ClientConfig(api_key_var="LOCAL_TEST_KEY", api_base_url="http://127.0.0.1:1/v1"),
            "fake-model",
            {"n": 1, "extra_body": {}},
        )
    )

    assert output["reward"] == 1.0
    assert output["accuracy"] == 1.0
    assert output["parseable_score"] == 1.0
    assert output["strict_format"] == 1.0
    assert output["info"] == {"dataset": "scifact"}
    assert output["answer"] == "SUPPORT"


@pytest.mark.skipif(
    os.environ.get("MEDFACT_BENCH_RUN_DATASET_INTEGRATION") != "1",
    reason="Set MEDFACT_BENCH_RUN_DATASET_INTEGRATION=1 to run the local Parquet integration test.",
)
def test_local_parquet_dataset_invariants(tmp_path: Path) -> None:
    if not REAL_DATASET_PATH.exists():
        pytest.fail(f"Configured MedFact-Bench Parquet file does not exist: {REAL_DATASET_PATH}")

    environment = load_environment(
        dataset_path=str(REAL_DATASET_PATH),
        cache_dir=str(tmp_path / "datasets-cache"),
    )
    eval_dataset = environment.eval_dataset

    assert len(eval_dataset) == 14_274
    assert Counter(row["info"]["dataset"] for row in eval_dataset) == Counter(EXPECTED_DATASET_COUNTS)
    assert Counter(row["answer"] for row in eval_dataset) == {
        "SUPPORT": 10_118,
        "NEI": 2_988,
        "CONTRADICT": 1_168,
    }
    assert environment.dataset is None
    assert all(len(row["prompt"]) == 2 for row in eval_dataset)
    assert all(row["prompt"][0]["role"] == "system" for row in eval_dataset)
    assert all(row["prompt"][1]["role"] == "user" for row in eval_dataset)
    assert all(row["question"] and row["answer"] and row["info"]["dataset"] for row in eval_dataset)

    duplicate_excess = sum(
        count - 1
        for count in Counter((row["question"], row["answer"], row["info"]["dataset"]) for row in eval_dataset).values()
        if count > 1
    )
    assert duplicate_excess == 2_261
