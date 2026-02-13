import asyncio

from datasets import Dataset

from environments.medcalc_bench.medcalc_bench.tools import SimpleToolEnv


def test_env_response_does_not_crash_on_malformed_tool_args() -> None:
    dummy = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    env = SimpleToolEnv(use_python=True, use_calculator=False, eval_dataset=dummy)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "code_interpreter", "arguments": "not json"},
                }
            ],
        }
    ]
    tool_messages = asyncio.run(env.env_response(messages, state={}))
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"
    assert "Invalid tool formatting" in tool_messages[0]["content"]


def test_env_response_treats_empty_args_as_empty_dict() -> None:
    dummy = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    env = SimpleToolEnv(use_python=True, use_calculator=False, eval_dataset=dummy)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "code_interpreter", "arguments": ""},
                }
            ],
        }
    ]
    tool_messages = asyncio.run(env.env_response(messages, state={}))
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"


def test_env_response_accepts_tool_calls_as_json_strings() -> None:
    dummy = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    env = SimpleToolEnv(use_python=True, use_calculator=False, eval_dataset=dummy)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                '{"id":"call_1","function":{"name":"code_interpreter","arguments":"not json"},"type":"function"}'
            ],
        }
    ]
    tool_messages = asyncio.run(env.env_response(messages, state={}))
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"
    assert "Invalid tool formatting" in tool_messages[0]["content"]


def test_env_response_accepts_double_encoded_arguments() -> None:
    dummy = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    env = SimpleToolEnv(use_python=True, use_calculator=False, eval_dataset=dummy)
    # arguments parses to a string, which itself is JSON
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "code_interpreter",
                        "arguments": '"{\\"code\\":\\"print(1)\\"}"',
                    },
                }
            ],
        }
    ]
    tool_messages = asyncio.run(env.env_response(messages, state={}))
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"
    assert tool_messages[0]["content"] == "1\n"


def test_env_response_accepts_dict_arguments() -> None:
    dummy = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    env = SimpleToolEnv(use_python=True, use_calculator=False, eval_dataset=dummy)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "code_interpreter",
                        "arguments": {"code": "print(2)"},
                    },
                }
            ],
        }
    ]
    tool_messages = asyncio.run(env.env_response(messages, state={}))
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"
    assert tool_messages[0]["content"] == "2\n"


def test_env_response_sanitizes_invalid_json_escapes_in_arguments() -> None:
    dummy = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    env = SimpleToolEnv(use_python=True, use_calculator=False, eval_dataset=dummy)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "code_interpreter",
                        # Invalid JSON escape: \q
                        "arguments": '{"code":"print(3)\\q"}',
                    },
                }
            ],
        }
    ]
    tool_messages = asyncio.run(env.env_response(messages, state={}))
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"
    assert "Invalid tool formatting" in tool_messages[0]["content"]
    # Ensure we don't resend invalid arguments on the next turn.
    assert messages[-1]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_env_response_drops_invalid_tool_names_from_message_history() -> None:
    dummy = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    env = SimpleToolEnv(use_python=True, use_calculator=False, eval_dataset=dummy)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        # Invalid name (contains spaces/newlines)
                        "name": "---\nLet's recompute",
                        "arguments": '{"code":"print(1)"}',
                    },
                }
            ],
        }
    ]
    tool_messages = asyncio.run(env.env_response(messages, state={}))
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"
    assert "Invalid tool formatting" in tool_messages[0]["content"]
    # The bad tool call must be rewritten to a known-good name/args, or some providers will 400 next turn.
    assert messages[-1]["tool_calls"][0]["function"]["name"] == "code_interpreter"
    assert messages[-1]["tool_calls"][0]["function"]["arguments"] == "{}"
