import json

import pytest

from medarc_verifiers.parsers import JSONParser


def test_parse_extracts_canonical_fields() -> None:
    parser = JSONParser(fields=["answer", "reasoning"])
    text = "The result is {\"answer\": \"42\", \"reasoning\": \"math\"}."

    parsed = parser.parse(text)

    assert isinstance(parsed, dict)
    assert parsed["answer"] == "42"
    assert parsed["reasoning"] == "math"


def test_parse_tracks_alternative_fields_independently() -> None:
    parser = JSONParser(fields=[("answer", "final_answer")], answer_field="answer")
    text = '{"final_answer": "correct", "answer": "redundant"}'

    parsed = parser.parse(text)

    assert isinstance(parsed, dict)
    assert parsed["answer"] == "redundant"
    assert parsed["final_answer"] == "correct"


def test_parse_stringifies_nested_structures() -> None:
    parser = JSONParser(fields=["answer", "metadata"])
    text = '{"answer": [1, 2, 3], "metadata": {"comment": "ok"}}'

    parsed = parser.parse(text)

    assert isinstance(parsed, dict)
    assert parsed["answer"] == [1, 2, 3]
    assert parsed["metadata"] == {"comment": "ok"}


def test_parse_returns_none_for_missing_object() -> None:
    parser = JSONParser(fields=["answer"])

    parsed = parser.parse("no json here")

    assert parsed is None


def test_parse_answer_prefers_canonical_then_alternatives() -> None:
    parser = JSONParser(fields=[("answer", "final_answer")], answer_field="answer")
    completion = [
        {"role": "assistant", "content": "garbage"},
        {"role": "assistant", "content": '{"final_answer": "alt value"}'},
    ]

    assert parser.parse_answer(completion) == "alt value"


def test_format_uses_canonical_field_names() -> None:
    parser = JSONParser(fields=[("answer", "final_answer"), "steps"])

    formatted = parser.format(final_answer="42", steps="show work")

    assert json.loads(formatted) == {"answer": "42", "steps": "show work"}


def test_reward_scores_parseability_and_field_coverage() -> None:
    parser = JSONParser(fields=[("answer", "final_answer"), "explanation"])
    reward = parser.get_format_reward_func()
    completion = [
        {
            "role": "assistant",
            "content": '{"answer": "ok", "explanation": "complete"}',
        }
    ]

    assert reward(completion) == pytest.approx(1.0)

    partial_completion = [
        {
            "role": "assistant",
            "content": '{"answer": "ok"}',
        }
    ]

    assert reward(partial_completion) == pytest.approx(0.75)


def test_pydantic_validation_failure_resets_fields_but_counts_parseable() -> None:
    pydantic = pytest.importorskip("pydantic")

    class Response(pydantic.BaseModel):
        answer: int

    parser = JSONParser(fields=["answer"], model=Response)
    reward = parser.get_format_reward_func()
    completion = [
        {
            "role": "assistant",
            "content": '{"answer": "not an int"}',
        }
    ]

    parsed = parser.parse('{"answer": "not an int"}')
    assert parsed is None
    assert reward(completion) == pytest.approx(0.5)
