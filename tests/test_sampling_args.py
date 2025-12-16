from medarc_verifiers.utils.sampling_args import sanitize_sampling_args_for_openai


def test_sanitize_sampling_args_merges_extra_body() -> None:
    result = sanitize_sampling_args_for_openai(
        {
            "temperature": 0.7,
            "extra_body": {"usage": {"include": True}},
            "top_k": 40,
        }
    )
    assert result["temperature"] == 0.7
    assert result["extra_body"]["usage"]["include"] is True
    assert result["extra_body"]["top_k"] == 40

