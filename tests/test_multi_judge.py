import asyncio

import pytest
from verifiers.rubrics.judge_rubric import JudgeRubric

from medarc_verifiers.judging.judge_cache_fix import install_cache_patch
from medarc_verifiers.judging.multi_judge import (
    MultiJudge,
    normalize_judge_endpoints,
    normalize_judge_models,
)
from medarc_verifiers.judging.multi_judge_rubric import MultiJudgeRubric


def test_normalize_judge_models():
    assert normalize_judge_models("gpt-4o-mini") == ["gpt-4o-mini"]
    assert normalize_judge_models(["a", "b"]) == ["a", "b"]
    assert normalize_judge_models([" gpt-4o ", "  claude-4.5  "]) == ["gpt-4o", "claude-4.5"]
    with pytest.raises(ValueError):
        normalize_judge_models([])
    with pytest.raises(ValueError):
        normalize_judge_models([""])


def test_normalize_judge_endpoints(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")
    base_urls, api_keys = normalize_judge_endpoints(None, None, 2)
    assert base_urls == [None, None]
    assert api_keys == ["secret", "secret"]

    base_urls, api_keys = normalize_judge_endpoints("https://api.openai.com/v1", None, 3)
    assert base_urls == ["https://api.openai.com/v1"] * 3
    assert len(api_keys) == 3

    base_urls, api_keys = normalize_judge_endpoints(["a"], None, 2)
    assert base_urls == ["a", "a"]
    assert api_keys == ["secret", "secret"]

    base_urls, api_keys = normalize_judge_endpoints(None, ["k1"], 2)
    assert base_urls == [None, None]
    assert api_keys == ["k1", "k1"]
    with pytest.raises(ValueError):
        normalize_judge_endpoints(["a", "b"], None, 3)
    with pytest.raises(ValueError):
        normalize_judge_endpoints(None, ["k1", "k2"], 3)


def test_aggregation_mean():
    assert MultiJudge.mean([1.0, None, 0.0]) == 0.5
    assert MultiJudge.mean([None, None]) == 0.0
    assert MultiJudge.mean([]) == 0.0


def test_multi_judge_rubric_injects_objects():
    mj = MultiJudge.__new__(MultiJudge)
    mj.judge = object()  # type: ignore[assignment]
    rubric = MultiJudgeRubric(mj)
    assert rubric.class_objects["judge"] is mj.judge
    assert rubric.class_objects["multi_judge"] is mj


@pytest.mark.asyncio
async def test_judge_cache_namespacing(monkeypatch: pytest.MonkeyPatch):
    install_cache_patch()
    monkeypatch.setattr("medarc_verifiers.utils.token_tracker.TOKEN_TRACKING_ENABLED", False, raising=False)

    calls = []

    async def fake_call_judge_model(judge_client, judge_model, judge_prompt, judge_sampling_args, logger):
        calls.append((judge_model, judge_prompt))
        return f"{judge_model}-ok", object()

    monkeypatch.setattr("medarc_verifiers.judging.judge_cache_fix.call_judge_model", fake_call_judge_model)

    class DummyClient:
        def __init__(self, base_url):
            self.base_url = base_url

    state = {}
    rubric_a = JudgeRubric(judge_client=DummyClient("http://a"), judge_model="model-a", judge_prompt="{question}")
    rubric_b = JudgeRubric(judge_client=DummyClient("http://b"), judge_model="model-b", judge_prompt="{question}")

    await rubric_a.judge("same", "", "", state)
    await rubric_b.judge("same", "", "", state)

    assert len(calls) == 2
    assert "http://a::model-a" in state["judge_response"]
    assert "http://b::model-b" in state["judge_response"]

    await rubric_a.judge("same", "", "", state)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_judge_cache_concurrent_writes(monkeypatch: pytest.MonkeyPatch):
    install_cache_patch()
    monkeypatch.setattr("medarc_verifiers.utils.token_tracker.TOKEN_TRACKING_ENABLED", False, raising=False)

    async def fake_call_judge_model(judge_client, judge_model, judge_prompt, judge_sampling_args, logger):
        if "openai" in str(judge_client.base_url):
            await asyncio.sleep(0.05)
            return "Response A", object()
        await asyncio.sleep(0.01)
        return "Response B", object()

    monkeypatch.setattr("medarc_verifiers.judging.judge_cache_fix.call_judge_model", fake_call_judge_model)

    class DummyClient:
        def __init__(self, base_url):
            self.base_url = base_url

    state = {}
    rubric_a = JudgeRubric(
        judge_client=DummyClient("https://api.openai.com/v1"), judge_model="gpt-4o", judge_prompt="{question}"
    )
    rubric_b = JudgeRubric(
        judge_client=DummyClient("https://api.anthropic.com/v1"), judge_model="claude-4.5", judge_prompt="{question}"
    )

    results = await asyncio.gather(
        rubric_a.judge("test prompt", "", "", state),
        rubric_b.judge("test prompt", "", "", state),
    )
    assert results == ["Response A", "Response B"]
    cache = state["judge_response"]
    assert cache["https://api.openai.com/v1::gpt-4o"]["test prompt"] == "Response A"
    assert cache["https://api.anthropic.com/v1::claude-4.5"]["test prompt"] == "Response B"


@pytest.mark.asyncio
async def test_multi_judge_results_order():
    call_order = []

    class StubRubric:
        def __init__(self, label, delay):
            self.label = label
            self.delay = delay

        async def judge(self, prompt, completion, answer, state):
            await asyncio.sleep(self.delay)
            call_order.append(self.label)
            return f"{self.label}-ok"

    mj = MultiJudge.__new__(MultiJudge)
    mj.judge_models = ["a", "b", "c"]
    mj.judge_base_urls = [None, None, None]
    mj.judge_ids = [f"default::{name}" for name in mj.judge_models]
    mj.judge_rubrics = [
        StubRubric("a", 0.03),
        StubRubric("b", 0.02),
        StubRubric("c", 0.01),
    ]

    results = await mj.judge("prompt", "", "", {})
    assert [result.raw for result in results] == ["a-ok", "b-ok", "c-ok"]
    assert call_order == ["c", "b", "a"]


@pytest.mark.asyncio
async def test_multi_judge_failure_handling():
    class StubRubric:
        def __init__(self, label, fail=False):
            self.label = label
            self.fail = fail

        async def judge(self, prompt, completion, answer, state):
            if self.fail:
                raise RuntimeError("Judge API error")
            return f"{self.label}-ok"

    mj = MultiJudge.__new__(MultiJudge)
    mj.judge_models = ["bad", "good"]
    mj.judge_base_urls = [None, None]
    mj.judge_ids = [f"default::{name}" for name in mj.judge_models]
    mj.judge_rubrics = [StubRubric("bad", fail=True), StubRubric("good", fail=False)]

    results = await mj.judge("prompt", "", "", {})
    assert results[0].raw is None
    assert "Judge API error" in results[0].error
    assert results[1].raw == "good-ok"
    assert results[1].error is None


@pytest.mark.asyncio
async def test_multi_judge_all_fail():
    class StubRubric:
        async def judge(self, prompt, completion, answer, state):
            raise RuntimeError("API error")

    mj = MultiJudge.__new__(MultiJudge)
    mj.judge_models = ["a", "b"]
    mj.judge_base_urls = [None, None]
    mj.judge_ids = [f"default::{name}" for name in mj.judge_models]
    mj.judge_rubrics = [StubRubric(), StubRubric()]

    results = await mj.judge("prompt", "", "", {})
    assert all(result.raw is None for result in results)
    assert all(result.error is not None for result in results)
