import verifiers as vf
from typing import Dict
from verifiers.utils.data_utils import extract_boxed_answer, THINK_BOXED_SYSTEM_PROMPT, BOXED_SYSTEM_PROMPT
from datasets import load_dataset, Dataset
from datasets.utils.logging import disable_progress_bar
import random
import re
disable_progress_bar() # suppress datasets mapping progress bar


def _build_question(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items() if v != "")
    return f"Question: {question}\nOptions:\n{opts}"


def _build_few_shot(few_shot_examples: Dataset, use_think: bool) -> str:
    # validation split used for few-shot examples, https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L173
    few_shot_prompt = ""
    for row in few_shot_examples:
        question = row["question"]
        cot = row["cot_content"]
        opts = row["options"]
        question_prompt = _build_question(question, opts)
        if use_think:
            few_shot_prompt += f"{question_prompt}\nAnswer:\n{cot}\n\n"
        else:
            # strip <think> tags if not using them
            cot_no_think = re.sub(r'<think>\s*', '', cot)
            cot_no_think = re.sub(r'\s*</think>', '', cot_no_think)
            few_shot_prompt += f"{question_prompt}\nAnswer:\n{cot_no_think}\n\n"
    return few_shot_prompt


def _to_vf_format(ds: Dataset, few_shot_examples: Dataset, shuffle: bool, use_think: bool) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': string the env will turn into chat messages
      - 'answer':   top-level gold letter (A/B/C/D[/E])
      - 'info':     keep all original fields for bookkeeping
    
    Args:
      - ds: dataset to convert to vf format
      - few_shot_examples: few-shot examples to include in the prompt
      - shuffle: whether to shuffle the answer choices
      - use_think: whether to use think tags
    """
    VALID = "ABCDEFGHIJ"
    
    def _format_row(row: dict) -> dict:
        question = row.get("question", "") or "" # question string
        opts = row.get("options", {}) or {} # answer choices, map of letter to answer text
        
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
            letters = VALID[:len(option_pairs)]
            opts = {letters[i]: text for i, (_, text) in enumerate(option_pairs)}
            
            # find the new letter for the correct answer
            for letter, text in opts.items():
                if text == correct_answer_text:
                    answer_letter = letter
                    break
        
        # https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L228
        instruction = "The following are multiple choice questions (with answers) about health. Think step by step and then output the single letter answer at the end like \\boxed{A}.\n\n" if use_think else ""
        few_shot_prompt = _build_few_shot(few_shot_examples, use_think)
        question_prompt = _build_question(question, opts)
        prompt = instruction + few_shot_prompt + question_prompt + "\nAnswer:"

        # question and answer have been moved to top-level, so remove them here
        info = dict(row)

        # update shuffled answer choices in the info dict
        if shuffle:
            info["answer"] = answer_letter
            info["options"] = opts

        return {
            "question": prompt,
            "answer": answer_letter,
            "info": info,
        }

    return ds.map(_format_row, remove_columns=ds.column_names).filter(lambda row: row is not None)


def load_environment(
        num_test_examples: int = -1, 
        num_few_shot: int = 1,
        use_think: bool = False,
        shuffle: bool = False,
        **kwargs
    ) -> vf.Environment:
    """
    Single-turn MMLU-Pro-Health environment using HuggingFace `mkieffer/MMLU-Pro-Health` dataset
    
    Each example is normalized to the fields expected by `vf.SingleTurnEnv`:
        {
            "question": "<stem + formatted options>",      # string used as the user prompt
            "answer":   "<A|B|C|D|...|J>",                     # top-level gold letter
            "info":     { ...original example fields... }  # full source row for debugging
        }

    - Parser extracts \\boxed{A|B|C|D|...|J} from completions

    - Reward looks for exact match between parsed letter and answer letter
    """

    # -------- load dataset --------
    # the validation split is used for few-shot examples, following the MMLU-Pro eval script
    test_raw, few_shot_examples = load_dataset("mkieffer/MMLU-Pro-Health", split=["test", "validation"])

    # -------- limit number of examples if specified --------
    if num_test_examples != -1:
        test_raw = test_raw.select(range(min(num_test_examples, len(test_raw))))
    if num_few_shot != -1:
        if num_few_shot > few_shot_examples.num_rows:    
            print(f"WARNING: num_few_shot={num_few_shot} is greater than the number of few-shot examples ({few_shot_examples.num_rows}). Using all examples.")
        few_shot_examples = few_shot_examples.select(range(min(num_few_shot, len(few_shot_examples))))

    # -------- convert rows to vf format and shuffle row order --------
    rng_seed = 12345
    few_shot_examples = few_shot_examples.shuffle(seed=rng_seed)
    test_ds = _to_vf_format(test_raw, few_shot_examples, shuffle=shuffle, use_think=use_think).shuffle(seed=rng_seed)
    del test_raw, few_shot_examples  # free memory
    
    # -------- construct prompts and questions --------
    parser = vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think else vf.Parser(extract_fn=extract_boxed_answer)
    system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT

    # -------- rubric --------
    def correct_answer_reward_func(parser, completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        response = response.strip()
        
        # remove \text{...} wrapper if present
        text_match = re.match(r'\\text\{(.+)\}', response)
        if text_match:
            response = text_match.group(1).strip()
        
        # try to extract a letter at the beginning
        # matches: "H", "H.", "H:", "(H)", "(H).", "H. Some text", "(A) Some text", etc.
        letter_match = re.match(r'^\(?([A-J])\)?(?:[.:\s]|$)', response)
        if letter_match:
            extracted_letter = letter_match.group(1)
            return 1.0 if extracted_letter == answer else 0.0
        
        # fallback to exact match
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
        # dataset=..., # no train dataset. val dataset used for few-shot examples 
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )