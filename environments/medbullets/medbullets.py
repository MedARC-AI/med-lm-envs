import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar() # suppress 'Generating...' messages from Dataset.from_generator

def _strip_E(split):
    for ex in split:
        ex = dict(ex)
        ex["options"] = {k: v for k, v in ex["options"].items() if k != "E"}
        yield ex

def _build_question_str(question: str, options: dict) -> str:
    s = f"Question: {question}\n"
    for k, v in options.items():
        # skip null values of v (for the combined dataset where E opt for 4op is null)
        if v is not None and v != "":
            s += f"\n{k}: {v}"
    return s

def _to_vf_format(ds: Dataset, split: str) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': string the env will turn into chat messages
      - 'answer':   top-level gold letter (A/B/C/D[/E])
      - 'info':     keep all original fields for bookkeeping
    """
    VALID = {"A","B","C","D","E"}

    def gen():
        for row in ds:
            row = dict(row)
            # build the user-visible question string (stem + options)
            q = row.get("question", "") or ""
            opts = row.get("options", {}) or {}
            question_str = _build_question_str(q, opts)

            # lift the answer to top-level, normalize to a single letter
            ans = (row.get("answer") or "").strip().upper()
            if ans not in VALID:
                # if op4 split sometimes stores 'E' or empty, coerce safely
                if ans == "" and "answer_letter" in row:
                    ans = str(row["answer_letter"]).strip().upper()
                if ans not in VALID:
                    # final guard: drop anything unexpected
                    ans = ""

            # keep full original example under 'info'
            info = dict(row)

            yield {
                "question": question_str,
                "answer": ans,
                "info": info,
            }

    return Dataset.from_generator(gen, split=split)

def load_environment(
        num_train_examples: int = -1, 
        num_eval_examples: int = -1,
        num_options: int = 4,
        use_think: bool = False,
        **kwargs
    ) -> vf.Environment:
    """
    Single-turn Medbullets environment using HuggingFace `mkieffer/Medbullets` dataset
    
    Each example is normalized to the fields expected by `vf.SingleTurnEnv`:
        {
            "question": "<stem + formatted options>",      # string used as the user prompt
            "answer":   "<A|B|C|D|E>",                     # top-level gold letter
            "info":     { ...original example fields... }  # full source row for debugging
        }

    - num_options=4 : loads splits `op4_train` / `op4_eval` and drops option "E"
    - num_options=5 : loads splits `op5_train` / `op5_eval`

    - Parser extracts \\boxed{A|B|C|D|E} from completions

    - Reward looks for exact match between parsed letter and answer letter
    """

    # -------- load dataset --------
    if num_options == 4:
        # 4 options: {"A", "B", "C", "D"}
        train_raw, eval_raw = load_dataset("mkieffer/Medbullets", split=["op4_train", "op4_eval"])
        # remove the "E" option from op4 splits
        train_raw = Dataset.from_generator(lambda: _strip_E(train_raw), split="op4_train")
        eval_raw  = Dataset.from_generator(lambda: _strip_E(eval_raw), split="op4_eval")
    elif num_options == 5:
        # 5 options: {"A", "B", "C", "D", "E"}
        train_raw, eval_raw = load_dataset("mkieffer/Medbullets", split=["op5_train", "op5_eval"])
    else: 
        raise ValueError("'num_options' must be 4 or 5")

    # -------- limit number of examples if specified --------
    if num_train_examples != -1:
        train_raw = train_raw.select(range(min(num_train_examples, len(train_raw))))
    if num_eval_examples != -1:
        eval_raw = eval_raw.select(range(min(num_eval_examples, len(eval_raw))))

    # -------- reshape to {'prompt', 'info'} --------
    rng_seed = 12345
    train_ds = _to_vf_format(train_raw, split="train").shuffle(seed=rng_seed)
    eval_ds  = _to_vf_format(eval_raw, split="eval").shuffle(seed=rng_seed)

    # -------- construct prompts and questions --------
    options = "(A, B, C, or D)" if num_options == 4 else "(A, B, C, D, or E)"

    if use_think:
        system_prompt = f"""Think step-by-step inside <think>...</think> tags, then give only the letter of the correct answer inside \\boxed{{...}} {options}. Do not include option text in the box; only the letter."""
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        system_prompt = f"""Give only the letter of the correct answer inside \\boxed{{...}} {options}. Do not include option text in the box; only the letter. /no_think"""
        parser = vf.Parser(extract_fn=extract_boxed_answer)

    # -------- rubric --------
    def correct_answer_reward_func(parser, completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        funcs=[
            correct_answer_reward_func,
            parser.get_format_reward_func()
        ],
        weights=[1.0, 0.0],
        parser=parser,
    )

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )