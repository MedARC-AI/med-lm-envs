"""Tests for randomize_multiple_choice utility function."""

import pytest

from medarc_verifiers.utils.randomize_mcq import randomize_multiple_choice


def test_no_seed_returns_unchanged_list():
    """With seed=None, options should not be shuffled."""
    opts = ["Option A", "Option B", "Option C"]
    labels = ["A", "B", "C"]
    result, new_label, new_idx = randomize_multiple_choice(opts, 0, labels=labels, seed=None)

    assert result == opts
    assert new_label == "A"
    assert new_idx == 0


def test_no_seed_returns_unchanged_dict():
    """With seed=None, dict options should not be shuffled."""
    opts = {"A": "Option A", "B": "Option B", "C": "Option C"}
    result, new_label, new_idx = randomize_multiple_choice(opts, "A", seed=None)

    assert result == opts
    assert new_label == "A"
    assert new_idx == 0


def test_deterministic_shuffle_list():
    """Same seed should produce same shuffle."""
    opts = ["Opt 1", "Opt 2", "Opt 3", "Opt 4", "Opt 5"]
    labels = ["A", "B", "C", "D", "E"]

    result1, label1, idx1 = randomize_multiple_choice(opts, 0, labels=labels, seed=42)
    result2, label2, idx2 = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    assert result1 == result2
    assert label1 == label2
    assert idx1 == idx2


def test_deterministic_shuffle_dict():
    """Same seed should produce same shuffle for dict."""
    opts = {"A": "Opt 1", "B": "Opt 2", "C": "Opt 3", "D": "Opt 4"}

    result1, label1, idx1 = randomize_multiple_choice(opts, "A", seed=42)
    result2, label2, idx2 = randomize_multiple_choice(opts, "A", seed=42)

    assert result1 == result2
    assert label1 == label2
    assert idx1 == idx2


def test_different_seed_different_shuffle():
    """Different seeds should produce different shuffles."""
    opts = ["Opt 1", "Opt 2", "Opt 3", "Opt 4", "Opt 5"]
    labels = ["A", "B", "C", "D", "E"]

    result1, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)
    result2, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=43)

    # With 5 options, extremely unlikely to get same shuffle
    assert result1 != result2


def test_row_id_affects_shuffle():
    """Different row_id should produce different shuffles."""
    opts = ["Opt 1", "Opt 2", "Opt 3", "Opt 4", "Opt 5"]
    labels = ["A", "B", "C", "D", "E"]

    result1, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42, row_id="row1")
    result2, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42, row_id="row2")

    # Different row_ids should give different shuffles
    assert result1 != result2


def test_non_deterministic_shuffle():
    """Seed=-1 should use non-deterministic randomness."""
    opts = ["Opt 1", "Opt 2", "Opt 3", "Opt 4", "Opt 5"]
    labels = ["A", "B", "C", "D", "E"]

    # Run multiple times - at least one should differ (probabilistic)
    results = []
    for _ in range(10):
        result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=-1)
        results.append(result)

    # Not all results should be identical
    unique_results = [list(r) for r in set(tuple(r) for r in results)]
    assert len(unique_results) > 1


def test_anchor_at_end_stays_fixed():
    """'All of the above' at end should not move."""
    opts = ["Option A", "Option B", "Option C", "All of the above"]
    labels = ["A", "B", "C", "D"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    # Last option should always be the anchor
    assert result[-1] == "All of the above"


def test_multiple_anchors_stay_fixed():
    """Multiple anchors should stay in their positions."""
    opts = ["Opt 1", "Opt 2", "Both of the above", "Opt 3", "All of the above"]
    labels = ["A", "B", "C", "D", "E"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    # Anchors should stay at indices 2 and 4
    assert result[2] == "Both of the above"
    assert result[4] == "All of the above"


def test_anchor_variations_detected():
    """Various anchor phrasings should be detected."""
    anchors = [
        "All of the above",
        "ALL OF THE ABOVE",
        "None of the above",
        "Some of the above",
        "Both of the above",
        "Neither of the above",
        "All of the following",
        "None of the following",
        "All of the above.",  # with punctuation
        "All of the above options",  # with suffix
    ]

    for anchor in anchors:
        opts = ["Opt 1", "Opt 2", anchor]
        labels = ["A", "B", "C"]
        result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

        # Anchor should stay at last position
        assert result[-1] == anchor, f"Failed for anchor: {anchor}"


def test_all_options_are_anchors():
    """When all options are anchors, nothing should shuffle."""
    opts = ["All of the above", "None of the above", "Both of the above"]
    labels = ["A", "B", "C"]

    result, _, _ = randomize_multiple_choice(opts, 1, labels=labels, seed=42)

    assert result == opts


def test_no_anchors_all_shuffle():
    """When no anchors, all options can shuffle."""
    opts = ["Option A", "Option B", "Option C"]
    labels = ["A", "B", "C"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    # Should be shuffled (with seed=42, unlikely to be same order)
    # Just verify it's a permutation
    assert sorted(result) == sorted(opts)


def test_answer_is_anchor():
    """When answer itself is an anchor, it should stay fixed."""
    opts = ["Option A", "Option B", "All of the above", "Option D"]
    labels = ["A", "B", "C", "D"]

    result, new_label, new_idx = randomize_multiple_choice(opts, 2, labels=labels, seed=42)

    # Anchor should stay at index 2
    assert new_idx == 2
    assert new_label == "C"
    assert result[2] == "All of the above"


# --- Answer tracking ---


def test_answer_index_tracked_list():
    """Answer index should be correctly updated after shuffle."""
    opts = ["Opt A", "Opt B", "Opt C", "All of the above"]
    labels = ["A", "B", "C", "D"]

    result, new_label, new_idx = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    # Find where "Opt A" moved
    assert result[new_idx] == "Opt A"
    # Verify label matches
    assert labels[new_idx] == new_label


def test_answer_index_tracked_dict():
    """Answer should be tracked correctly for dict input."""
    opts = {"A": "Opt 1", "B": "Opt 2", "C": "Opt 3"}

    result, new_label, new_idx = randomize_multiple_choice(opts, "A", seed=42)

    # The label at new_idx should point to "Opt 1"
    result_list = list(result.values())
    assert result_list[new_idx] == "Opt 1"


def test_answer_string_label_parsed():
    """String labels like 'B' should be parsed correctly."""
    opts = {"A": "Opt 1", "B": "Opt 2", "C": "Opt 3"}

    result, new_label, new_idx = randomize_multiple_choice(opts, "B", seed=42)

    # Should find "Opt 2"
    result_list = list(result.values())
    assert result_list[new_idx] == "Opt 2"


def test_labels_with_parentheses():
    """Labels like '(A)', '(B)' should work."""
    opts = {"(A)": "Opt 1", "(B)": "Opt 2", "(C)": "Opt 3"}

    result, new_label, new_idx = randomize_multiple_choice(opts, "(A)", seed=42)

    result_list = list(result.values())
    assert result_list[new_idx] == "Opt 1"


def test_numeric_labels():
    """Numeric labels like '1.', '2.' should work."""
    opts = {"1.": "Opt 1", "2.": "Opt 2", "3.": "Opt 3"}

    result, new_label, new_idx = randomize_multiple_choice(opts, "1.", seed=42)

    result_list = list(result.values())
    assert result_list[new_idx] == "Opt 1"


def test_lowercase_labels():
    """Lowercase labels should work."""
    opts = {"a": "Opt 1", "b": "Opt 2", "c": "Opt 3"}

    result, new_label, new_idx = randomize_multiple_choice(opts, "a", seed=42)

    result_list = list(result.values())
    assert result_list[new_idx] == "Opt 1"


def test_answer_as_integer_index():
    """Integer answer_choice should work."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]
    labels = ["A", "B", "C"]

    result, new_label, new_idx = randomize_multiple_choice(opts, 1, labels=labels, seed=42)

    # Should track "Opt 2"
    assert result[new_idx] == "Opt 2"


def test_list_input_returns_list():
    """List input should return list."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]
    labels = ["A", "B", "C"]

    result, new_label, new_idx = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    assert isinstance(result, list)
    assert isinstance(new_label, str)
    assert isinstance(new_idx, int)


def test_dict_input_returns_dict():
    """Dict input should return dict."""
    opts = {"A": "Opt 1", "B": "Opt 2", "C": "Opt 3"}

    result, new_label, new_idx = randomize_multiple_choice(opts, "A", seed=42)

    assert isinstance(result, dict)
    assert isinstance(new_label, str)
    assert isinstance(new_idx, int)


def test_list_with_no_seed_returns_label_not_index():
    """List mode with seed=None should return label string, not int."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]
    labels = ["A", "B", "C"]

    result, new_label, new_idx = randomize_multiple_choice(opts, 1, labels=labels, seed=None)

    # Should return label "B", not integer 1
    assert new_label == "B"
    assert isinstance(new_label, str)
    assert new_idx == 1


def test_invalid_answer_choice_raises():
    """Invalid answer_choice should raise ValueError."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]
    labels = ["A", "B", "C"]

    with pytest.raises(ValueError, match="not found or invalid"):
        randomize_multiple_choice(opts, "Z", labels=labels, seed=42)


def test_missing_labels_for_list_raises():
    """List input without labels should raise ValueError."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]

    with pytest.raises(ValueError, match="labels must be provided"):
        randomize_multiple_choice(opts, 0, seed=42)


def test_out_of_range_index_raises():
    """Out of range integer answer should raise ValueError."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]
    labels = ["A", "B", "C"]

    with pytest.raises(ValueError, match="out of range"):
        randomize_multiple_choice(opts, 10, labels=labels, seed=42)


def test_single_option():
    """Single option should work without error."""
    opts = ["Only option"]
    labels = ["A"]

    result, new_label, new_idx = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    assert result == opts
    assert new_label == "A"
    assert new_idx == 0


def test_two_options_both_anchors():
    """Two options that are both anchors."""
    opts = ["All of the above", "None of the above"]
    labels = ["A", "B"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    assert result == opts


def test_empty_option_text():
    """Empty strings in options should not crash."""
    opts = ["Opt 1", "", "Opt 3"]
    labels = ["A", "B", "C"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    # Should complete without error
    assert len(result) == 3


def test_dict_preserves_label_order():
    """Dict labels should stay in their positions."""
    opts = {"X": "Opt 1", "Y": "Opt 2", "Z": "Opt 3"}

    result, _, _ = randomize_multiple_choice(opts, "X", seed=42)

    # Labels should be unchanged
    assert list(result.keys()) == ["X", "Y", "Z"]
    # But values may be shuffled
    assert sorted(result.values()) == sorted(opts.values())


def test_row_id_can_be_integer():
    """row_id as integer should work."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]
    labels = ["A", "B", "C"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42, row_id=123)

    # Should complete without error
    assert len(result) == 3


def test_row_id_can_be_string():
    """row_id as string should work."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]
    labels = ["A", "B", "C"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42, row_id="question_1")

    # Should complete without error
    assert len(result) == 3
