"""Tests for the open_i_summarization environment."""

import pytest


def test_extract_completion_text_from_list() -> None:
    """Test extracting text from chat-style completion."""
    from open_i_summarization import _extract_completion_text

    completion = [{"role": "assistant", "content": "No acute findings"}]
    assert _extract_completion_text(completion) == "No acute findings"


def test_extract_completion_text_from_string() -> None:
    """Test extracting text from string completion."""
    from open_i_summarization import _extract_completion_text

    completion = "No acute findings"
    assert _extract_completion_text(completion) == "No acute findings"


def test_extract_completion_text_empty_list() -> None:
    """Test handling empty completion list."""
    from open_i_summarization import _extract_completion_text

    completion = []
    assert _extract_completion_text(completion) == "[]"


def test_extract_answer_section_with_think_tags() -> None:
    """Test extracting answer after think tags."""
    from open_i_summarization import extract_answer_section

    text = "<think>Some reasoning here...</think>The actual answer"
    assert extract_answer_section(text) == "The actual answer"


def test_extract_answer_section_without_think_tags() -> None:
    """Test extracting answer without think tags."""
    from open_i_summarization import extract_answer_section

    text = "Just a plain answer"
    assert extract_answer_section(text) == "Just a plain answer"


def test_extract_answer_section_empty() -> None:
    """Test handling empty string."""
    from open_i_summarization import extract_answer_section

    assert extract_answer_section("") == ""


def test_compute_normalized_judge_reward_perfect_scores() -> None:
    """Test normalized reward with perfect scores."""
    from open_i_summarization import _compute_normalized_judge_reward

    scores = {
        "correctness": {"score": 5, "reason": "Perfect"},
        "completeness": {"score": 5, "reason": "Perfect"},
        "conciseness": {"score": 5, "reason": "Perfect"},
    }
    assert _compute_normalized_judge_reward(scores) == pytest.approx(1.0)


def test_compute_normalized_judge_reward_mixed_scores() -> None:
    """Test normalized reward with mixed scores."""
    from open_i_summarization import _compute_normalized_judge_reward

    scores = {
        "correctness": {"score": 5, "reason": "Good"},
        "completeness": {"score": 3, "reason": "Partial"},
        "conciseness": {"score": 4, "reason": "Good"},
    }
    # (5/5 + 3/5 + 4/5) / 3 = (1.0 + 0.6 + 0.8) / 3 = 0.8
    assert _compute_normalized_judge_reward(scores) == pytest.approx(0.8)


def test_compute_normalized_judge_reward_zero_scores() -> None:
    """Test normalized reward with zero scores."""
    from open_i_summarization import _compute_normalized_judge_reward

    scores = {
        "correctness": {"score": 0, "reason": "Poor"},
        "completeness": {"score": 0, "reason": "Poor"},
        "conciseness": {"score": 0, "reason": "Poor"},
    }
    assert _compute_normalized_judge_reward(scores) == pytest.approx(0.0)


def test_compute_normalized_judge_reward_missing_scores() -> None:
    """Test normalized reward with missing scores."""
    from open_i_summarization import _compute_normalized_judge_reward

    scores = {
        "correctness": {"score": 5, "reason": "Good"},
        # completeness and conciseness missing
    }
    # Only correctness is counted: (5/5) / 3 = 0.333...
    assert _compute_normalized_judge_reward(scores) == pytest.approx(1 / 3)


def test_compute_normalized_judge_reward_string_scores() -> None:
    """Test normalized reward handles string scores."""
    from open_i_summarization import _compute_normalized_judge_reward

    scores = {
        "correctness": {"score": "4", "reason": "Good"},
        "completeness": {"score": "3", "reason": "Partial"},
        "conciseness": {"score": "5", "reason": "Excellent"},
    }
    # (4/5 + 3/5 + 5/5) / 3 = (0.8 + 0.6 + 1.0) / 3 = 0.8
    assert _compute_normalized_judge_reward(scores) == pytest.approx(0.8)


def test_judge_template_contains_placeholders() -> None:
    """Test that judge template has required placeholders."""
    from open_i_summarization import JUDGE_TEMPLATE

    assert "{findings}" in JUDGE_TEMPLATE
    assert "{response}" in JUDGE_TEMPLATE
    assert "{reference}" in JUDGE_TEMPLATE
    assert "{output_format}" in JUDGE_TEMPLATE


def test_judge_dimensions_defined() -> None:
    """Test that judge dimensions are properly defined."""
    from open_i_summarization import JUDGE_DIMENSIONS

    # Note: "correctness" replaces "accuracy" per Nature Medicine paper terminology
    assert "correctness" in JUDGE_DIMENSIONS
    assert "completeness" in JUDGE_DIMENSIONS
    assert "conciseness" in JUDGE_DIMENSIONS
    assert len(JUDGE_DIMENSIONS) == 3


def test_dataset_loading() -> None:
    """Test that the dataset can be loaded."""
    from datasets import load_dataset

    ds = load_dataset("medarc/open-i-summarization", split="test")
    assert len(ds) == 343
    assert "idx" in ds.features
    assert "inputs" in ds.features
    assert "target" in ds.features


def test_dataset_sample_structure() -> None:
    """Test sample structure from dataset."""
    from datasets import load_dataset

    ds = load_dataset("medarc/open-i-summarization", split="test")
    sample = ds[0]

    assert isinstance(sample["idx"], int)
    assert isinstance(sample["inputs"], str)
    assert isinstance(sample["target"], str)
    assert len(sample["inputs"]) > 0
    assert len(sample["target"]) > 0


@pytest.fixture
def mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a mock API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


def test_environment_loading(mock_api_key: None) -> None:
    """Test that environment loads successfully with mock API key."""
    import verifiers as vf

    env = vf.load_environment("open_i_summarization", compute_auto_metrics=False)

    assert env.name == "open_i_summarization"
    assert len(env.eval_dataset) == 343
    assert "Summarize the radiology report findings" in env.system_prompt


def test_environment_dataset_mapping(mock_api_key: None) -> None:
    """Test that dataset is properly mapped to environment format."""
    import verifiers as vf

    env = vf.load_environment("open_i_summarization", compute_auto_metrics=False)
    sample = env.eval_dataset[0]

    assert "question" in sample
    assert "answer" in sample
    assert "info" in sample
    assert "findings" in sample["info"]
    assert "idx" in sample["info"]


def test_environment_with_validation_split(mock_api_key: None) -> None:
    """Test loading validation split."""
    import verifiers as vf

    env = vf.load_environment("open_i_summarization", split="validation", compute_auto_metrics=False)
    assert len(env.eval_dataset) == 341


def test_environment_with_train_split(mock_api_key: None) -> None:
    """Test loading train split."""
    import verifiers as vf

    env = vf.load_environment("open_i_summarization", split="train", compute_auto_metrics=False)
    assert len(env.eval_dataset) == 2735
