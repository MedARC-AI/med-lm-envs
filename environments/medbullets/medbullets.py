import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, THINK_BOXED_SYSTEM_PROMPT, BOXED_SYSTEM_PROMPT
from datasets import load_dataset, Dataset
from datasets.utils.logging import disable_progress_bar
import random
disable_progress_bar() # suppress datasets mapping progress bar


def _build_question_str(question: str, options: dict) -> str:
    s = f"Question: {question}\n"
    for k, v in options.items():
        # skip null values of v (for the combined dataset where E opt for 4op is null)
        if v is not None and v != "":
            s += f"\n{k}: {v}"
    return s


def _to_vf_format(ds: Dataset, split: str, num_options: int, shuffle: bool) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': string the env will turn into chat messages
      - 'answer':   top-level gold letter (A/B/C/D[/E])
      - 'info':     keep all original fields for bookkeeping
    
    Args:
      - num_options: 4 or 5; if 4, strips out option "E"
      - shuffle: whether to shuffle the answer choices
    """
    VALID = {"A","B","C","D","E"}

    def _format_row(row: dict) -> dict:
        question = row.get("question", "") or "" # question string
        opts = row.get("options", {}) or {} # answer choices, map of letter to answer text
        
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
            letters = ["A", "B", "C", "D", "E"][:len(option_pairs)]
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


def load_environment(
        num_train_examples: int = -1, 
        num_eval_examples: int = -1,
        num_options: int = 4,
        use_think: bool = False,
        shuffle: bool = False,
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

    # -------- convert rows to vf format and shuffle row order --------
    rng_seed = 12345
    train_ds = _to_vf_format(train_raw, split="train", num_options=num_options, shuffle=shuffle).shuffle(seed=rng_seed)
    eval_ds  = _to_vf_format(eval_raw, split="eval", num_options=num_options, shuffle=shuffle).shuffle(seed=rng_seed)
    del train_raw, eval_raw  # free memory

    # -------- construct prompts and questions --------
    if use_think:
        system_prompt = THINK_BOXED_SYSTEM_PROMPT
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        system_prompt = BOXED_SYSTEM_PROMPT
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