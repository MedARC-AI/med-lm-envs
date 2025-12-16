"""Tests for the simplified MCQ accuracy grader."""

import pytest

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


def test_last_token_disabled_when_explicit_anchor_exists_even_if_wrong():
    # Regression: do NOT allow last_token to override an explicit (wrong) anchored choice.
    response = (
        "The correct answer is D. Mouth breathing.\n\nA and B are incorrect because ...\n\nC is incorrect because ..."
    )
    assert not multiple_choice_accuracy(response, answer_letter="C", answer_text="Both")


def test_last_token_ignores_negative_context_incorrect():
    # Regression: option tokens in contexts like "C is incorrect" should not count as the chosen answer.
    response = "No anchor here. C is incorrect because the appliance is not used for both."
    assert not multiple_choice_accuracy(response, answer_letter="C", answer_text="Both", accept_answer_text=False)


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


def test_unpaired_think_close_then_final_token_newline():
    # Regression: some models emit </think> without a matching <think> and place the answer after it.
    response = "Reasoning...\n</think>\n\nA"
    assert multiple_choice_accuracy(response, answer_letter="A", answer_text="Option A")


def test_unpaired_think_spurious_match():
    # Regression: some models emit </think> without a matching <think> and place the answer after it.
    response = "The answer is B. But on the other hand... I know\n</think>\n\nA"
    assert multiple_choice_accuracy(response, answer_letter="A", answer_text="Option A")


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


def test_direct_answer_method_details():
    result = multiple_choice_accuracy("C", answer_letter="C", answer_text="Option C", return_details=True)
    assert result.is_correct is True
    assert result.method == "direct_answer"
    assert result.matched_answer == "c"
    assert result.correct_answer == "C"


def test_prefix_priority_over_generic_anchor():
    # The explicit prefix match should be used even if a later generic anchor appears
    response = "Answer: B. The answer is C."
    result = multiple_choice_accuracy(
        response, answer_letter="B", answer_text="Option B", prefix="Answer:", return_details=True
    )
    assert result.is_correct is True
    assert result.method == "anchored_token"
    assert result.matched_answer == "B"


def test_last_token_negation_in_previous_sentence_not_blocking():
    # Negation in an earlier sentence should not block a later correct token
    response = "Not C. However, after reconsideration, the answer is actually C"
    assert multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C")


def test_leading_two_digit_option():
    assert multiple_choice_accuracy("12) Cranial nerve XII", answer_letter="12", answer_text="CN XII")


def test_answer_text_whitespace_flexibility():
    assert multiple_choice_accuracy(
        "The correct diagnosis is acute   kidney\tinjury.", answer_letter="A", answer_text="acute kidney injury"
    )


def test_invalid_answer_letter_raises():
    with pytest.raises(ValueError):
        multiple_choice_accuracy("Answer: C", answer_letter="AA", answer_text="Option C")


def test_anchored_negated_with_parentheses_should_fail():
    assert not multiple_choice_accuracy("Final answer is not (C).", answer_letter="C", answer_text="Option C")


def test_anchored_negated_isnt_ascii_apostrophe_should_fail():
    assert not multiple_choice_accuracy("The answer isn't C; it's D.", answer_letter="C", answer_text="Option C")


def test_anchored_negated_isnt_curly_apostrophe_should_fail():
    assert not multiple_choice_accuracy("The answer isn’t C; it’s D.", answer_letter="C", answer_text="Option C")


def test_anchored_negated_prefix_style_should_fail():
    assert not multiple_choice_accuracy("Answer: not C", answer_letter="C", answer_text="Option C")


def test_prefix_anchored_negated_should_fail():
    assert not multiple_choice_accuracy(
        "The answer is: not C", answer_letter="C", answer_text="Option C", prefix="The answer is:"
    )


def test_anchored_not_after_option_should_not_block():
    # Current implementation only treats "not/isn't" as negation when it appears BEFORE the option token.
    # So "Answer: C, not D" should still count as C.
    assert multiple_choice_accuracy("Answer: C, not D.", answer_letter="C", answer_text="Option C")


def test_leading_option_with_no_answer_text_should_pass():
    # Regression: ensure "No" as answer text doesn't trigger negation handling
    assert multiple_choice_accuracy("B. No", answer_letter="B", answer_text="No")


def test_leading_option_with_no_and_punctuation_should_pass():
    assert multiple_choice_accuracy("B) No.", answer_letter="B", answer_text="No")


def test_last_token_negation_same_sentence_blocks():
    # No anchor phrase, so it falls to last_token.
    # Because "Not" is in the same sentence, the final "C" should be blocked.
    assert not multiple_choice_accuracy("Not C, wait, C", answer_letter="C", answer_text="Option C")


def test_last_token_negation_previous_sentence_does_not_block():
    # Sentence boundary prevents earlier negation from blocking later token.
    assert multiple_choice_accuracy("Not C. C", answer_letter="C", answer_text="Option C")


def test_last_token_isnt_previous_sentence_does_not_block():
    assert multiple_choice_accuracy("It isn't C. C", answer_letter="C", answer_text="Option C")


def test_last_token_isnt_same_sentence_blocks():
    assert not multiple_choice_accuracy("It isn't C, but maybe C", answer_letter="C", answer_text="Option C")


def test_answer_text_does_not_override_explicit_wrong_choice():
    response = (
        "The other options do not account for the renal findings as well:\n"
        "- **D (Protein deposition):** Suggests amyloidosis...\n\n"
        "Therefore, the most accurate explanation is ... → choice **B**"
    )
    assert not multiple_choice_accuracy(response, answer_letter="D", answer_text="Protein deposition")


def test_explicit_choice_correct_even_with_other_option_texts_present():
    response = "- **D (Protein deposition):** ...\nAnswer: B"
    assert multiple_choice_accuracy(
        response, answer_letter="B", answer_text="Immune response to streptococcal infection"
    )


def test_answer_text_used_when_no_explicit_choice_letter_present():
    response = (
        "The presentation is classic for poststreptococcal glomerulonephritis.\n"
        "Therefore the diagnosis is poststreptocococcal glomerulonephritis."
    )
    assert multiple_choice_accuracy(response, answer_letter="B", answer_text="poststreptocococcal glomerulonephritis")


def test_negated_anchor_does_not_block_answer_text_fallback():
    response = "The answer is not C. The correct diagnosis is acute appendicitis."
    assert multiple_choice_accuracy(response, answer_letter="D", answer_text="acute appendicitis")


@pytest.mark.parametrize(
    "response",
    [
        "**B.** No",
        "*B.* No",
        "__B.__ No",
        "`B.` No",
        "~~B.~~ No",
        "- **B.** No",
        "* **B.** No",
        "+ **B.** No",
        "1. **B.** No",
        "12. **B.** No",
        "> **B)** No",
        "   >   -   **B.**   No",
        "**(B).** No",
        "__((B))).__ No",
    ],
)
def test_leading_option_markdown_wrapped_letter(response):
    assert multiple_choice_accuracy(response, answer_letter="B", answer_text="No")


@pytest.mark.parametrize(
    "response, letter, text",
    [
        ("**12)** Cranial nerve XII", "12", "Cranial nerve XII"),
        ("> 12) Cranial nerve XII", "12", "Cranial nerve XII"),
        ("- `2:` Second option", "2", "Second option"),
    ],
)
def test_leading_option_markdown_wrapped_number(response, letter, text):
    assert multiple_choice_accuracy(response, answer_letter=letter, answer_text=text)


def test_leading_option_markdown_returns_anchored_token_method():
    result = multiple_choice_accuracy("**B.** No", answer_letter="B", answer_text="No", return_details=True)
    assert result.is_correct is True
    assert result.method == "anchored_token"
    assert result.matched_answer == "B"
    assert result.correct_answer == "B"


@pytest.mark.parametrize(
    "response",
    [
        "**No.**",  # no leading option token
        "*No*",  # no delimiter after a letter/number token
        "- **No**",  # list prefix but no option token
        "> - No",  # quote/list but no option token
    ],
)
def test_leading_option_markdown_non_option_does_not_force_leading_match(response):
    # We just assert it doesn't incorrectly grade B as correct purely from leading parsing.
    assert not multiple_choice_accuracy(response, answer_letter="B", answer_text="No", accept_answer_text=False)


def test_not_leading_markdown_answer_is_not_matched_by_leading_strategy():
    response = "Answer: **B.** No"
    result = multiple_choice_accuracy(
        response,
        answer_letter="B",
        answer_text="No",
        accept_answer_text=False,
        return_details=True,
    )
    assert result.is_correct is True
    assert result.method == "anchored_token"
    assert result.matched_answer == "B"


def test_block_prompt_then_option_on_next_line_parses_choice_letter():
    response = "The most appropriate next step is:\n\n**(B) Video-capsule endoscopy**"
    assert multiple_choice_accuracy(
        response, answer_letter="B", answer_text="Video-capsule endoscopy", accept_answer_text=False
    )


def test_anchor_phrase_with_markdown_wrapper_parses_choice_letter():
    response = "Answer: **(C)**"
    result = multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C", return_details=True)
    assert result.is_correct is True
    assert result.method == "anchored_token"
    assert result.matched_answer == "C"


# --- Additional formatting coverage ---


@pytest.mark.parametrize(
    "response",
    [
        "Final answer — C",  # em dash
        "Final answer – C",  # en dash
        "Final answer —(C)",
        "Answer:（C）",  # fullwidth parentheses
        "Answer：C",  # fullwidth colon
        "Answer → C",  # unicode arrow
        "Answer ⇒ C",
        "Answer: 【C】",  # CJK brackets
    ],
)
def test_unicode_punctuation_variants_still_find_choice(response):
    # We only assert correctness; method may vary (anchored vs last_token).
    assert multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C", accept_answer_text=False)


@pytest.mark.parametrize(
    "response",
    [
        "I think OptionC is correct",  # letter embedded in a word
        "This is about C++ programming",  # C followed by '+'
        "B-cell lymphoma is mentioned here",  # letter followed by '-'
        "A/B testing is common in software",  # letters adjacent to '/'
    ],
)
def test_letter_embedded_in_longer_tokens_is_not_treated_as_choice(response):
    # These should NOT be graded as selecting the choice letter.
    assert not multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C", accept_answer_text=False)


@pytest.mark.parametrize(
    "response",
    [
        "Respuesta: C",  # Spanish
        "Réponse : C",  # French
        "Antwort: C",  # German
        "Risposta: C",  # Italian
    ],
)
def test_non_english_with_colon_still_works_via_token_extraction(response):
    # We don't have non-English anchors; but the letter token is still extractable.
    assert multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C", accept_answer_text=False)


@pytest.mark.parametrize(
    "response",
    [
        "答案是C",  # Chinese: no separator between word chars and 'C'
        "정답은C",  # Korean: same issue
        "答えはC",  # Japanese: same issue
    ],
)
def test_non_english_without_separator_does_not_match_choice_letter(response):
    # Without a separator, the option letter is embedded in a larger token.
    assert not multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C", accept_answer_text=False)


@pytest.mark.parametrize(
    "response, answer_text",
    [
        ("The diagnosis is Parkinson's disease", "Parkinson disease"),
        ("Organism is C. difficile", "C difficile"),
        ("Treat with TNF-alpha inhibitor", "TNF alpha inhibitor"),
        ("Use beta-blocker", "beta blocker"),
    ],
)
def test_answer_text_requires_exact_formatting_beyond_normalization(response, answer_text):
    # We only allow whitespace/case/unicode normalization; punctuation differences should not match.
    assert not multiple_choice_accuracy(response, answer_letter="D", answer_text=answer_text, accept_answer_text=True)


def test_multiple_answers_last_explicit_anchor_wins():
    response = "Answer: B. After reconsideration, final answer: C"
    assert multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C")


def test_multiple_answers_last_explicit_anchor_wins_even_if_first_was_correct():
    response = "Answer: C. Actually I change my mind: Answer: B"
    assert multiple_choice_accuracy(response, answer_letter="B", answer_text="Option B")
    assert not multiple_choice_accuracy(response, answer_letter="C", answer_text="Option C")


def test_multiple_answers_negated_then_corrected():
    response = "Answer: C. Not C. Final answer: B"
    assert multiple_choice_accuracy(response, answer_letter="B", answer_text="Option B")


def test_solution5_answer_text_matches_beginning_region():
    llm_answer = "Thoracic aortic rupture is the correct diagnosis based on the clinical presentation."
    assert multiple_choice_accuracy(
        llm_answer=llm_answer,
        answer_letter="C",
        answer_text="Thoracic aortic rupture",
        accept_answer_text=True,
    )


def test_solution5_answer_text_matches_end_region_after_reasoning():
    llm_answer = """
    The patient presents with severe chest trauma. After analyzing all symptoms,
    including hemodynamic instability, widened mediastinum, and mechanism of injury,
    we can conclude the diagnosis is:

    Thoracic aortic rupture
    """
    assert multiple_choice_accuracy(
        llm_answer=llm_answer,
        answer_letter="C",
        answer_text="Thoracic aortic rupture",
        accept_answer_text=True,
    )


def test_solution5_answer_text_does_not_match_only_in_middle_reasoning():
    padding_before = " ".join(["word"] * 120)
    padding_after = " ".join(["text"] * 120)
    llm_answer = (
        padding_before
        + " While thoracic aortic rupture is a consideration, it is not definitive. "
        + padding_after
        + " The actual answer is: Tension pneumothorax"
    )
    assert not multiple_choice_accuracy(
        llm_answer=llm_answer,
        answer_letter="C",
        answer_text="Thoracic aortic rupture",
        accept_answer_text=True,
    )


def test_solution5_answer_text_allows_not_inside_answer_text():
    long_answer = "Respect the patient's prior wishes and do not resuscitate"
    llm_answer = f"Based on the advance directive, the correct action is: {long_answer}"
    assert multiple_choice_accuracy(
        llm_answer=llm_answer,
        answer_letter="B",
        answer_text=long_answer,
        accept_answer_text=True,
    )


def test_solution5_answer_text_blocked_when_negated_before_match_in_sentence():
    llm_answer = "The diagnosis is not Thoracic aortic rupture."
    assert not multiple_choice_accuracy(
        llm_answer=llm_answer,
        answer_letter="C",
        answer_text="Thoracic aortic rupture",
        accept_answer_text=True,
    )


def test_solution5_answer_text_blocked_when_negation_precedes_location_phrase():
    llm_answer = "Brown adipose tissue is most likely not found in Scapula."
    assert not multiple_choice_accuracy(
        llm_answer=llm_answer,
        answer_letter="A",
        answer_text="Scapula",
        accept_answer_text=True,
    )


def test_solution5_parkinson_substring_anchor_regression():
    result = multiple_choice_accuracy(
        llm_answer="Parkinson disease",
        answer_letter="E",
        answer_text="Parkinson disease",
        accept_answer_text=True,
        return_details=True,
    )
    assert result.is_correct is True
    assert result.method == "answer_text"
