import pytest

from medarc_verifiers.parsers import XMLParser


def test_parse_messages_returns_last_with_tags() -> None:
    parser = XMLParser(["think", "answer"])
    completion = [
        {"role": "assistant", "content": "no tags here"},
        {"role": "assistant", "content": "<answer>early</answer>"},
        {"role": "assistant", "content": "<think>trace</think>\n<answer>final</answer>"},
    ]

    parsed = parser.parse(completion, last=True)

    assert parsed is not None
    assert parsed.answer == "final"
    assert parsed.think == "trace"


def test_parse_string_handles_tags() -> None:
    parser = XMLParser(["think", "answer"])

    parsed = parser.parse("<think>inner</think><answer>42</answer>")

    assert parsed is not None
    assert parsed.answer == "42"
    assert parsed.think == "inner"


def test_init_with_think_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        XMLParser(["think", "answer"])

    assert all("think" not in record.message for record in caplog.records)
