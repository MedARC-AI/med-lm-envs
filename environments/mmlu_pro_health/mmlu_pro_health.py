import random
import re
from typing import Dict

import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets mapping progress bar


def _build_question(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items() if v not in [None, ""])
    return f"Question: {question}\nOptions:\n{opts}"


def _normalize_cot_answer(text: str) -> str:
    """Normalize explicit answer phrasing to boxed format."""
    if "boxed{" in text:
        return text
    return re.sub(r"The answer is \(([A-J])\)", r"The answer is \\boxed{\1}", text)


def _process_age_strings(text: str, rng: random.Random) -> str:
    """Add a small decimal jitter (~±2 weeks) to ages from M-ARC."""
    age_pattern = re.compile(
        r"\b(\d+)"  # numeric age
        r"(?:\s*[-\s]?)"
        r"(year-old|year old|yo)\b",
        re.IGNORECASE,
    )

    def add_decimal(match: re.Match[str]) -> str:
        age = int(match.group(1))
        decimal = rng.uniform(0, 0.04)
        new_age = age + decimal
        return f"{new_age:.3f} year-old"

    return age_pattern.sub(add_decimal, text)


def _build_few_shot(few_shot_examples: Dataset, use_think: bool) -> str:
    # validation split used for few-shot examples, https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L173
    few_shot_prompt = ""
    for row in few_shot_examples:
        question = row["question"]
        cot = row["cot_content"]
        opts = row["options"]
        question_prompt = _build_question(question, opts)
        if use_think:
            few_shot_prompt += f"{question_prompt}\nAnswer:\n{_normalize_cot_answer(cot)}\n\n"
        else:
            # strip <think> tags if not using them
            cot_no_think = re.sub(r"<think>\s*", "", cot)
            cot_no_think = re.sub(r"\s*</think>", "", cot_no_think)
            few_shot_prompt += f"{question_prompt}\nAnswer:\n{_normalize_cot_answer(cot_no_think)}\n\n"
    return few_shot_prompt


def _to_vf_format(
    ds: Dataset,
    few_shot_examples: Dataset,
    shuffle_answers: bool,
    shuffle_seed: int | None,
    use_think: bool,
    jitter_age: bool,
) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': string the env will turn into chat messages
      - 'answer':   top-level gold letter (A/B/C/D[/E])
      - 'info':     keep all original fields for bookkeeping

    Args:
      - ds: dataset to convert to vf format
      - few_shot_examples: few-shot examples to include in the prompt
      - shuffle_answers: whether to shuffle the answer choices
      - shuffle_seed: deterministic seed for choice shuffling
      - use_think: whether to use think tags
      - jitter_age: whether to add a small decimal jitter (~±2 weeks) to ages
    """
    VALID = "ABCDEFGHIJ"

    def _format_row(row: dict, idx: int) -> dict:
        question = row.get("question", "") or ""  # question string
        opts = row.get("options", {}) or {}  # answer choices, map of letter to answer text
        opts = {k: v for k, v in opts.items() if v not in [None, ""]}

        # lift the answer to top-level, normalize to a single letter
        answer_letter = (row.get("answer") or "").strip().upper()
        if answer_letter not in VALID:
            return None

        # shuffle answer choices if requested
        if shuffle_answers and answer_letter and answer_letter in opts:
            opts, answer_letter, _ = randomize_multiple_choice(
                options=opts,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=idx,
            )

        # https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L228
        instruction = 'The following are multiple choice questions (with answers) about health. Think step by step and then finish your answer with "\\boxed{X}" where X is the correct letter choice.\n'
        few_shot_prompt = _build_few_shot(few_shot_examples, use_think)
        # Per-row RNG keeps age jitter deterministic and process-safe for num_proc>1
        base_seed = shuffle_seed if (shuffle_answers and shuffle_seed is not None) else 2718
        question_text = _process_age_strings(question, random.Random(base_seed + idx)) if jitter_age else question
        question_prompt = _build_question(question_text, opts)
        prompt = instruction + few_shot_prompt + question_prompt + "\nAnswer:"

        # question and answer have been moved to top-level, so remove them here
        info = dict(row)

        # update shuffled answer choices in the info dict
        if shuffle_answers:
            info["answer"] = answer_letter
            info["options"] = opts
        info["answer_text"] = opts.get(answer_letter, None)

        return {"question": prompt, "answer": answer_letter, "info": info}

    return ds.map(_format_row, remove_columns=ds.column_names, load_from_cache_file=False, with_indices=True)


def load_environment(
    num_few_shot: int = 5,
    use_think: bool = False,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    jitter_age: bool = False,
    **kwargs,
) -> vf.Environment:
    """
    Single-turn MMLU-Pro-Health environment using HuggingFace `TIGER-Lab/MMLU-Pro` dataset

    Args:
        num_few_shot (int): Number of few-shot examples to include in the prompt.
        use_think (bool): Whether to use think tags and think parser.
        shuffle_answers (bool): Whether to shuffle the answer choices.
        shuffle_seed (int or None): Deterministic seed for choice shuffling.
        jitter_age (bool): Whether to add a small decimal jitter (~±2 weeks) to ages from M-ARC.
    """

    # -------- load dataset --------
    # the validation split is used for few-shot examples, following the MMLU-Pro eval script
    test_raw, few_shot_examples = [
        ds.filter(lambda row: row["category"] == "health")
        for ds in load_dataset("TIGER-Lab/MMLU-Pro", split=["test", "validation"])
    ]

    # -------- convert options from list to dict --------
    # MMLU-Pro options are stored as a list, convert to dict with letter keys
    def _convert_options(row: dict) -> dict:
        opts = row["options"]
        if isinstance(opts, list):
            row["options"] = {chr(ord("A") + i): v for i, v in enumerate(opts) if v.strip().upper() != "N/A"}
        return row

    test_raw = test_raw.map(_convert_options, load_from_cache_file=False)
    few_shot_examples = few_shot_examples.map(_convert_options, load_from_cache_file=False)

    # -------- limit number of examples if specified --------
    if num_few_shot != -1:
        if num_few_shot > few_shot_examples.num_rows:
            print(
                f"WARNING: num_few_shot={num_few_shot} is greater than the number of few-shot examples ({few_shot_examples.num_rows}). Using all examples."
            )
        few_shot_examples = few_shot_examples.select(range(min(num_few_shot, len(few_shot_examples))))

    # -------- convert rows to vf format and shuffle row order --------
    few_shot_examples = few_shot_examples
    test_ds = _to_vf_format(
        test_raw,
        few_shot_examples,
        shuffle_answers=shuffle_answers,
        shuffle_seed=shuffle_seed,
        use_think=use_think,
        jitter_age=jitter_age,
    )
    del test_raw, few_shot_examples  # free memory

    # -------- construct prompts and questions --------
    parser = (
        vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think else vf.Parser(extract_fn=extract_boxed_answer)
    )
    system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT

    # -------- rubric --------
    def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        # dataset=..., # no train dataset. val dataset used for few-shot examples
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
