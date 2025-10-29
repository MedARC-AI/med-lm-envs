"""Tests for the simplified MCQ accuracy grader."""

from medarc_verifiers.rewards.multiple_choice_accuracy import MCQAccuracyResult, multiple_choice_accuracy


def test_anchored_final_answer_colon():
    assert multiple_choice_accuracy("Let me think... Final answer: C", answer_letter="C", answer_text="Correct option")


def test_anchored_the_answer_is():
    assert multiple_choice_accuracy("After analysis, the answer is D.", answer_letter="D", answer_text="Another option")


def test_anchored_therefore_style():
    assert multiple_choice_accuracy(
        "Based on the evidence, therefore B is correct.", answer_letter="B", answer_text="Some text"
    )


def test_anchored_with_parentheses():
    assert multiple_choice_accuracy("My choice: (A)", answer_letter="A", answer_text="Option A")


def test_anchored_negated_should_fail():
    assert not multiple_choice_accuracy("The answer is not C, it's D.", answer_letter="C", answer_text="Wrong option")


def test_anchored_wrong_letter():
    assert not multiple_choice_accuracy("Final answer: B", answer_letter="C", answer_text="Correct option")


def test_anchored_numeric():
    assert multiple_choice_accuracy("The answer is 3", answer_letter="3", answer_text="Third option")


def test_last_token_single_letter_at_end():
    assert multiple_choice_accuracy("I think it's C", answer_letter="C", answer_text="Correct option")


def test_last_token_with_period():
    assert multiple_choice_accuracy("My selection is B.", answer_letter="B", answer_text="Some text")


def test_last_token_multiple_letters_takes_last():
    # A and B appear in reasoning, D is the final answer
    assert multiple_choice_accuracy("A is wrong. B seems unlikely. D", answer_letter="D", answer_text="Final option")


def test_last_token_negated_should_fail():
    assert not multiple_choice_accuracy("Definitely not C!", answer_letter="C", answer_text="Wrong option")


def test_last_token_numeric():
    assert multiple_choice_accuracy("I choose option 2", answer_letter="2", answer_text="Second option")


def test_last_token_wrong():
    assert not multiple_choice_accuracy("My answer is A", answer_letter="B", answer_text="Correct")


def test_answer_text_exact_match():
    assert multiple_choice_accuracy(
        "The correct treatment is chemotherapy and radiation",
        answer_letter="C",
        answer_text="chemotherapy and radiation",
    )


def test_answer_text_in_sentence():
    assert multiple_choice_accuracy(
        "Based on the symptoms, acute myocardial infarction is most likely.",
        answer_letter="B",
        answer_text="acute myocardial infarction",
    )


def test_answer_text_case_insensitive():
    assert multiple_choice_accuracy(
        "The diagnosis is DIABETES MELLITUS TYPE 2", answer_letter="D", answer_text="Diabetes Mellitus Type 2"
    )


def test_answer_text_with_negation_fails():
    assert not multiple_choice_accuracy(
        "This is not hypertension, it's hypotension.", answer_letter="A", answer_text="hypertension"
    )


def test_answer_text_disabled():
    assert not multiple_choice_accuracy(
        "The answer is hypertension", answer_letter="A", answer_text="hypertension", accept_answer_text=False
    )


def test_answer_text_substring_not_matched():
    # "tension" should not match "hypertension"
    assert not multiple_choice_accuracy("Patient has tension headaches", answer_letter="A", answer_text="hypertension")


def test_normalization_extra_whitespace():
    assert multiple_choice_accuracy("Final   answer:    C  ", answer_letter="C", answer_text="Option C")


def test_normalization_unicode():
    assert multiple_choice_accuracy(
        "The answer is C",  # Different unicode space
        answer_letter="C",
        answer_text="Option C",
    )


def test_normalization_prefix_stripping():
    assert multiple_choice_accuracy(
        "The answer is: C", answer_letter="C", answer_text="Option C", prefix="The answer is:"
    )


def test_normalization_empty_answer():
    assert not multiple_choice_accuracy("", answer_letter="C", answer_text="Option C")


def test_normalization_latex_stripping():
    # Test basic LaTeX removal (if pylatexenc available)
    assert multiple_choice_accuracy(
        r"The answer is \textbf{C}", answer_letter="C", answer_text="Option C", strip_tex=True
    )


def test_return_details_anchored_token():
    result = multiple_choice_accuracy("Final answer: C", answer_letter="C", answer_text="Option C", return_details=True)
    assert isinstance(result, MCQAccuracyResult)
    assert result.is_correct is True
    assert result.method == "anchored_token"
    assert result.matched_answer == "C"
    assert result.correct_answer == "C"


def test_return_details_last_token():
    result = multiple_choice_accuracy("I think it's B", answer_letter="B", answer_text="Option B", return_details=True)
    assert isinstance(result, MCQAccuracyResult)
    assert result.is_correct is True
    assert result.method == "last_token"
    assert result.matched_answer == "B"
    assert result.correct_answer == "B"


def test_return_details_answer_text():
    result = multiple_choice_accuracy(
        "The patient has acute appendicitis", answer_letter="D", answer_text="acute appendicitis", return_details=True
    )
    assert isinstance(result, MCQAccuracyResult)
    assert result.is_correct is True
    assert result.method == "answer_text"
    assert result.matched_answer == "the patient has acute appendicitis"
    assert result.correct_answer == "acute appendicitis"


def test_return_details_no_match():
    result = multiple_choice_accuracy("I don't know", answer_letter="C", answer_text="Option C", return_details=True)
    assert isinstance(result, MCQAccuracyResult)
    assert result.is_correct is False
    assert result.method == "none"
    assert result.matched_answer is None


def test_return_details_bool_by_default():
    result = multiple_choice_accuracy("Answer: C", answer_letter="C", answer_text="Option C", return_details=False)
    assert isinstance(result, bool)
    assert result is True


def test_cot_with_anchored_final_answer():
    cot_response = """
    Let me analyze each option:

    A) This is incorrect because the patient's symptoms don't match.
    B) This could be possible, but the timeline doesn't fit.
    C) This seems most likely given the presentation.
    D) This is ruled out by the lab results.

    Final answer: C
    """
    assert multiple_choice_accuracy(
        cot_response,
        answer_letter="C",
        answer_text="Correct diagnosis",
    )


def test_cot_without_anchor_uses_last_token():
    cot_response = """
    Considering the options:
    - A is wrong due to age
    - B doesn't fit symptoms
    - C matches perfectly

    C
    """
    assert multiple_choice_accuracy(cot_response, answer_letter="C", answer_text="Match")


def test_cot_prevents_early_letter_matching():
    # Should not match A or B from the reasoning
    cot_response = """
    A) Incorrect - patient is too young
    B) Possible but unlikely
    C) Most likely diagnosis

    The answer is C
    """
    assert multiple_choice_accuracy(cot_response, answer_letter="C", answer_text="Likely diagnosis")


def test_edge_case_multiple_parenthetical_options():
    assert multiple_choice_accuracy(
        "Could be (A) or (B), but I choose (C)",
        answer_letter="C",
        answer_text="Final choice",
    )


def test_edge_case_letter_in_medical_term():
    # "C" in "Vitamin C" should not be matched as answer
    assert multiple_choice_accuracy(
        "Patient needs Vitamin C supplementation. Answer: D", answer_letter="D", answer_text="Supplement"
    )


def test_edge_case_hemoglobin_a1c():
    # "A" in "A1c" should not match
    assert multiple_choice_accuracy("HbA1c is elevated. The answer is B", answer_letter="B", answer_text="Option B")


def test_edge_case_decimal_numbers():
    # Should not break on decimals
    assert multiple_choice_accuracy(
        "Level is 3.5 mg/dL. Answer: A",
        answer_letter="A",
        answer_text="Normal range",
    )


def test_edge_case_mixed_case_letter():
    assert multiple_choice_accuracy("The answer is c", answer_letter="C", answer_text="Option")


def test_edge_case_multiline_answer():
    assert multiple_choice_accuracy("Reasoning...\n\nFinal answer:\nC", answer_letter="C", answer_text="Option C")


def test_edge_case_quoted_answer_text():
    assert multiple_choice_accuracy(
        'The diagnosis is "acute bronchitis"', answer_letter="B", answer_text="acute bronchitis"
    )
