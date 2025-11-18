import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets mapping progress bar


def _build_question_str(question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question: {question}\n\n{opts}"


def _to_vf_format(
    ds: Dataset,
    num_options: int,
    shuffle_answers: bool,
    shuffle_seed: int | None,
) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': string the env will turn into chat messages
      - 'answer':   top-level gold letter (A/B/C/D[/E])
      - 'info':     keep all original fields for bookkeeping

    Args:
      - num_options: 4 or 5; if 4, strips out option "E"
      - shuffle_answers: whether to shuffle the answer choices
      - shuffle_seed: deterministic seed forwarded to the shuffler
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
        if shuffle_answers and answer_letter and answer_letter in opts:
            shuffled_options, answer_letter, _ = randomize_multiple_choice(
                options=opts,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=question,
            )
            opts = shuffled_options

        question_str = _build_question_str(question, opts)

        # question and answer have been moved to top-level, so remove them here
        info = dict(row)

        # update shuffled answer choices in the info dict
        if shuffle_answers:
            info["answer"] = answer_letter
            info["options"] = opts
        info["answer_text"] = opts[answer_letter]

        return {
            "question": question_str,
            "answer": answer_letter,
            "answer_text": opts[answer_letter],
            "info": info,
        }

    # Disable the Datasets cache when shuffling answers
    load_from_cache_file = False if shuffle_answers else True
    mapped = ds.map(
        _format_row,
        remove_columns=ds.column_names,
        load_from_cache_file=load_from_cache_file,
    )
    return mapped.filter(lambda row: row is not None, load_from_cache_file=load_from_cache_file)


def load_environment(
    num_options: int = 4,
    use_think: bool = False,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.BOXED,
    **kwargs,
) -> vf.Environment:
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

    test_ds = _to_vf_format(
        test_raw, num_options=num_options, shuffle_answers=shuffle_answers, shuffle_seed=shuffle_seed
    )
    del test_raw  # free memory

    # normalize answer_format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    if answer_format == AnswerFormat.XML:
        system_prompt = THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        parser = (
            vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think else vf.Parser(extract_fn=extract_boxed_answer)
        )
        system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(eval_dataset=test_ds, system_prompt=system_prompt, parser=parser, rubric=rubric, **kwargs)
