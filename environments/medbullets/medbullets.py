import random

import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets mapping progress bar


def _build_question_str(question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question: {question}\n\n{opts}"


def _to_vf_format(ds: Dataset, num_options: int, shuffle: bool) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': string the env will turn into chat messages
      - 'answer':   top-level gold letter (A/B/C/D[/E])
      - 'info':     keep all original fields for bookkeeping

    Args:
      - num_options: 4 or 5; if 4, strips out option "E"
      - shuffle: whether to shuffle the answer choices
    """
    VALID = ("A", "B", "C", "D", "E")

    def _format_row(row: dict) -> dict:
        question = row.get("question", "") or ""  # question string
        opts = row.get("options", {}) or {}  # answer choices, map of letter to answer text

        # strip option E if num_options == 4
        if num_options == 4:
            opts = {k: v for k, v in opts.items() if k != "E"}

        # lift the answer to top-level, normalize to a single letter
        answer_letter = (row.get("answer") or "").strip().upper()
        if answer_letter not in VALID:
            return None

        # shuffle answer choices if requested
        if shuffle and answer_letter and answer_letter in opts:
            # get the correct answer text before shuffling
            correct_answer_text = opts[answer_letter]

            # create list of (letter, text) pairs and shuffle them
            option_pairs = list(opts.items())

            # use a deterministic seed based on the question for consistency
            rng = random.Random(hash(question) % (2**32))
            rng.shuffle(option_pairs)

            # rebuild options dict with new letter assignments
            letters = VALID[: len(option_pairs)]
            opts = {letters[i]: text for i, (_, text) in enumerate(option_pairs)}

            # find the new letter for the correct answer
            for letter, text in opts.items():
                if text == correct_answer_text:
                    answer_letter = letter
                    break

        question_str = _build_question_str(question, opts)

        # question and answer have been moved to top-level, so remove them here
        info = dict(row)

        # update shuffled answer choices in the info dict
        if shuffle:
            info["answer"] = answer_letter
            info["options"] = opts

        return {
            "question": question_str,
            "answer": answer_letter,
            "info": info,
        }

    return ds.map(_format_row, remove_columns=ds.column_names).filter(lambda row: row is not None)


def load_environment(num_options: int = 4, use_think: bool = False, shuffle: bool = False, **kwargs) -> vf.Environment:
    """
    Single-turn Medbullets environment using HuggingFace `mkieffer/Medbullets` dataset

    Each example is normalized to the fields expected by `vf.SingleTurnEnv`:
        {
            "question": "<stem + formatted options>",      # string used as the user prompt
            "answer":   "<A|B|C|D|E>",                     # top-level gold letter
            "info":     { ...original example fields... }  # full source row for debugging
        }

    - num_options=4 : loads split `op4_test
    - num_options=5 : loads split `op5_test`

    - Parser extracts \\boxed{A|B|C|D|E} from completions

    - Reward looks for exact match between parsed letter and answer letter
    """

    # -------- load dataset --------
    if num_options == 4:
        # 4 options: {"A", "B", "C", "D"}
        test_raw = load_dataset("mkieffer/Medbullets", split="op4_test")
    elif num_options == 5:
        # 5 options: {"A", "B", "C", "D", "E"}
        test_raw = load_dataset("mkieffer/Medbullets", split="op5_test")
    else:
        raise ValueError("'num_options' must be 4 or 5")

    test_ds = _to_vf_format(test_raw, num_options=num_options, shuffle=shuffle)
    del test_raw  # free memory

    parser = (
        vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think else vf.Parser(extract_fn=extract_boxed_answer)
    )
    system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT

    def correct_answer_reward_func(parser, completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func],
        weights=[1.0],
        parser=parser,
    )

    return vf.SingleTurnEnv(eval_dataset=test_ds, system_prompt=system_prompt, parser=parser, rubric=rubric, **kwargs)
