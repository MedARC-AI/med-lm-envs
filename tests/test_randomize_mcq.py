"""Tests for randomize_multiple_choice utility function."""

import pytest

from medarc_verifiers.utils import randomize_multiple_choice_hf_map, randomize_multiple_choice_row
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice


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
        "None of the choices are correct",
        "All answers are correct",
        "Some statements are true",
        "Both options apply",
        "None apply",
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


def test_labels_required_for_list():
    """List input without labels should raise ValueError."""
    opts = ["Opt 1", "Opt 2", "Opt 3"]

    with pytest.raises(ValueError, match="labels must be provided"):
        randomize_multiple_choice(opts, 0, seed=42)


def test_labels_not_required_for_dict():
    """Dict input doesn't need labels parameter."""
    opts = {"A": "Opt 1", "B": "Opt 2", "C": "Opt 3", "D": "Opt 4"}

    result, new_label, new_idx = randomize_multiple_choice(opts, "D", seed=None)

    # With no shuffle, answer D should stay at position 3 with label D
    assert new_label == "D"
    assert new_idx == 3


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


def test_consecutive_anchors_preserve_order():
    """Consecutive anchors should remain in their original order and positions."""
    opts = [
        "Option 1",
        "All of the above",
        "None of the above",
        "Option 4",
    ]
    labels = ["A", "B", "C", "D"]

    result, _, _ = randomize_multiple_choice(opts, 0, labels=labels, seed=42)

    # Anchors should stay exactly where they were
    assert result[1] == "All of the above"
    assert result[2] == "None of the above"
    # Non-anchor blocks are: [0] and [3], which are length-1 and won't move anyway


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


# --- HuggingFace datasets helper function ---


def test_randomize_multiple_choice_hf_map_basic():
    """randomize_multiple_choice_hf_map should work with basic example dict."""

    example = {"options": ["Opt 1", "Opt 2", "Opt 3"], "answer": 0}

    result = randomize_multiple_choice_hf_map(example, idx=0, seed=42)

    assert "options" in result
    assert "answer" in result
    assert "answer_label" in result
    assert isinstance(result["options"], list)
    assert isinstance(result["answer"], int)
    assert isinstance(result["answer_label"], str)


def test_randomize_multiple_choice_hf_map_with_index():
    """randomize_multiple_choice_hf_map should use idx as row_id for determinism."""

    example = {"options": ["Opt 1", "Opt 2", "Opt 3", "Opt 4"], "answer": 0}

    result1 = randomize_multiple_choice_hf_map(example, idx=0, seed=42)
    result2 = randomize_multiple_choice_hf_map(example, idx=0, seed=42)
    result3 = randomize_multiple_choice_hf_map(example, idx=1, seed=42)

    # Same idx should give same shuffle
    assert result1["options"] == result2["options"]
    # Different idx should give different shuffle
    assert result1["options"] != result3["options"]


def test_randomize_multiple_choice_hf_map_without_index():
    """randomize_multiple_choice_hf_map should work without idx (uses hash)."""

    example = {"options": ["Opt 1", "Opt 2", "Opt 3"], "answer": 1}

    result = randomize_multiple_choice_hf_map(example, seed=42)

    # Should still work
    assert result["options"][result["answer"]] == "Opt 2"


def test_randomize_multiple_choice_hf_map_custom_keys():
    """randomize_multiple_choice_hf_map should accept custom field names."""

    example = {"choices": ["A", "B", "C"], "correct": 1}

    result = randomize_multiple_choice_hf_map(example, idx=0, seed=42, options_key="choices", answer_key="correct")

    assert "choices" in result
    assert "correct" in result
    assert result["choices"][result["correct"]] == "B"


def test_randomize_multiple_choice_hf_map_no_label():
    """randomize_multiple_choice_hf_map should skip answer_label if return_label=False."""

    example = {"options": ["Opt 1", "Opt 2", "Opt 3"], "answer": 0}

    result = randomize_multiple_choice_hf_map(example, idx=0, seed=42, return_label=False)

    assert "answer_label" not in result
    assert "options" in result
    assert "answer" in result


def test_randomize_multiple_choice_hf_map_preserves_anchors():
    """randomize_multiple_choice_hf_map should preserve anchor options."""

    example = {"options": ["Opt 1", "Opt 2", "All of the above"], "answer": 0}

    result = randomize_multiple_choice_hf_map(example, idx=0, seed=42)

    # Anchor should stay at end
    assert result["options"][-1] == "All of the above"


def test_randomize_multiple_choice_hf_map_hf_map_compatible():
    """randomize_multiple_choice_hf_map return format should be map-compatible."""

    example = {"question": "Q1", "options": ["A", "B", "C"], "answer": 1, "metadata": {"id": 123}}

    # Simulate what map does: merge result into example
    result = randomize_multiple_choice_hf_map(example, idx=0, seed=42)

    # Result should only have the keys it needs to update
    assert set(result.keys()) == {"options", "answer", "answer_label"}

    # In real map, you'd do: return {**example, **result} or just return result
    # Let's verify the pattern works
    updated = {**example, **result}
    assert updated["question"] == "Q1"  # Preserved
    assert updated["metadata"] == {"id": 123}  # Preserved
    assert "answer_label" in updated  # Added


def test_randomize_multiple_choice_hf_map_answer_as_label():
    """Setting answer_as_index=False should keep the answer as a label."""

    example = {"options": ["Opt 1", "Opt 2", "Opt 3"], "answer": 0}

    result = randomize_multiple_choice_hf_map(example, idx=0, seed=42, answer_as_index=False)

    assert isinstance(result["answer"], str)
    assert result["answer"] == result["answer_label"]


# ========================================
# Row Helper Tests
# ========================================


def test_randomize_multiple_choice_row_dict_options():
    """Row helper should shuffle dict options and update answer_text."""

    row = {
        "options": {"A": "Opt 1", "B": "Opt 2", "C": "Opt 3"},
        "answer": "A",
        "answer_text": "Opt 1",
        "extra": 123,
    }

    updated = randomize_multiple_choice_row(row, seed=42)

    assert updated is not row
    assert set(updated["options"].keys()) == {"A", "B", "C"}
    assert updated["answer"] in updated["options"]
    assert updated["answer_text"] == updated["options"][updated["answer"]]
    # Ensure unrelated fields preserved
    assert updated["extra"] == 123


def test_randomize_multiple_choice_row_return_mapping():
    """Row helper should optionally return the permutation mapping."""

    row = {"options": ["Opt 1", "Opt 2", "Opt 3"], "answer": 1}

    (updated, mapping) = randomize_multiple_choice_row(
        row,
        seed=42,
        return_mapping=True,
    )

    assert isinstance(mapping, list)
    assert updated["options"][mapping.index(1)] == "Opt 2"
    assert updated["answer"] == chr(ord("A") + mapping.index(1))
    assert updated["answer_text"] == updated["options"][mapping.index(1)]


# ========================================
# Anchor Pattern Detection Tests
# ========================================


def test_traditional_anchor_pattern_detection():
    """ANCHOR regex should detect 'all/none of the above' patterns."""
    from medarc_verifiers.utils.randomize_multiple_choice import ANCHOR

    anchors = [
        "All of the above",
        "None of the above",
        "All of the following",
        "None of the following",
        "Both of the above",
        "Neither of the above",
        "Some of the above",
    ]
    for text in anchors:
        assert ANCHOR.search(text), f"Should match: {text}"


def test_label_reference_pattern_detection():
    """LABEL_REF regex should detect options that reference other labels."""
    from medarc_verifiers.utils.randomize_multiple_choice import LABEL_REF

    label_refs = [
        "A or B",
        "A and B",
        "Both A and B",
        "Either A or B",
        "Neither A nor B",
        "A, B, or C",
        "A & B",
        "A/B",
        "(A) or (B)",
        "A) and B)",
        "[A] or [B]",
        "Only A and B",
    ]
    for text in label_refs:
        assert LABEL_REF.search(text), f"Should match: {text}"


def test_label_reference_pattern_no_false_positives():
    """LABEL_REF regex should not match normal medical text."""
    from medarc_verifiers.utils.randomize_multiple_choice import LABEL_REF

    non_refs = [
        "Vitamin A deficiency",
        "Hepatitis B",
        "Type A personality",
        "Class A drug",
        "Schedule B medication",
        "The patient has condition A",
        "Treatment involves A and careful monitoring",
        "Administer drug A",
    ]
    for text in non_refs:
        assert not LABEL_REF.search(text), f"Should NOT match: {text}"


def test_traditional_anchor_stays_in_place():
    """'All of the above' should stay at the end when shuffling."""
    options = ["Option 1", "Option 2", "Option 3", "All of the above"]
    labels = ["A", "B", "C", "D"]

    shuffled, new_label, new_idx = randomize_multiple_choice(
        options=options,
        answer_choice=0,  # Answer is "Option 1"
        labels=labels,
        seed=42,
    )

    # "All of the above" should still be at position D (index 3)
    assert shuffled[3] == "All of the above"
    assert labels[3] == "D"


def test_label_reference_skips_shuffling():
    """Questions with label references like 'A or B' should not be shuffled."""
    options = ["Diabetes", "Hypertension", "A or B", "Obesity"]
    labels = ["A", "B", "C", "D"]

    shuffled, new_label, new_idx = randomize_multiple_choice(
        options=options,
        answer_choice=0,  # Answer is "Diabetes"
        labels=labels,
        seed=42,
    )

    # Should return completely unchanged (no shuffling occurred)
    assert shuffled == options
    assert new_label == "A"
    assert new_idx == 0


def test_multiple_traditional_anchors_stay_in_place():
    """Multiple 'X of the above' anchors should all stay fixed."""
    options = ["Option 1", "None of the above", "Option 3", "All of the above"]
    labels = ["A", "B", "C", "D"]

    shuffled, new_label, new_idx = randomize_multiple_choice(
        options=options,
        answer_choice=0,  # Answer is "Option 1"
        labels=labels,
        seed=42,
    )

    # Both anchors should stay in their original positions
    assert shuffled[1] == "None of the above"
    assert shuffled[3] == "All of the above"


def test_only_non_anchors_shuffle():
    """Only non-anchor options should get shuffled."""
    options = ["Diabetes", "Hypertension", "Obesity", "Hyperlipidemia", "Smoking", "All of the above"]
    labels = ["A", "B", "C", "D", "E", "F"]

    # Run multiple shuffles to check variability
    results = []
    for seed_val in range(10, 110, 20):
        shuffled, _, _ = randomize_multiple_choice(
            options=options,
            answer_choice=0,
            labels=labels,
            seed=seed_val,
        )
        results.append(shuffled)

        # Anchor should always stay in place
        assert shuffled[5] == "All of the above"

    # At least some variation in the non-anchor positions
    # Check first 5 positions for variation
    first_block_sets = [tuple(r[i] for i in range(5)) for r in results]
    # With 5 items and 5 different seeds, we should see at least 2 different orderings
    assert len(set(first_block_sets)) >= 2, f"Options should show variation, got: {first_block_sets}"


def test_anchors_create_multiple_shuffle_blocks():
    """Anchors should divide options into separate shuffle blocks."""
    options = [
        "Heart failure",
        "Diabetes",
        "None of the above",  # Anchor - divides into blocks
        "Obesity",
        "Hypertension",
        "All of the above",  # Anchor
    ]
    labels = ["A", "B", "C", "D", "E", "F"]

    shuffled, new_label, new_idx = randomize_multiple_choice(
        options=options,
        answer_choice=0,  # Answer is "Heart failure"
        labels=labels,
        seed=42,
    )

    # Anchors stay in place
    assert shuffled[2] == "None of the above"
    assert shuffled[5] == "All of the above"

    # First block (indices 0-1) can shuffle
    assert set(shuffled[0:2]) == {"Heart failure", "Diabetes"}

    # Second block (indices 3-4) can shuffle
    assert set(shuffled[3:5]) == {"Obesity", "Hypertension"}


def test_label_ref_skip_preserves_answer_tracking():
    """When shuffling is skipped due to label refs, answer tracking should still work."""
    options = ["Diabetes", "Hypertension", "Both A and B", "Obesity"]
    labels = ["A", "B", "C", "D"]

    # Test with answer being the label reference itself
    shuffled, new_label, new_idx = randomize_multiple_choice(
        options=options,
        answer_choice=2,  # Answer is "Both A and B"
        labels=labels,
        seed=42,
    )

    # Nothing should change
    assert shuffled == options
    assert new_label == "C"
    assert new_idx == 2
