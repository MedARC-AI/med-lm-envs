from typing import Dict

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice


def _build_prompt(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    letters = ", ".join(sorted(options.keys()))
    return (
        "You are a clinician. Choose exactly ONE option letter.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"Answer with ONLY the letter ({letters})."
    )


def load_environment(
    split: str = "test",
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    **kwargs,
) -> vf.Environment:
    """
    MetaMedQA multiple-choice accuracy eval
    - Loads HF 'maximegmd/MetaMedQA'
    - Builds a prompt per item
    - Scores accuracy by comparing the first A–Z from the model to the gold letter
    """
    ds = load_dataset("maximegmd/MetaMedQA", split=split)

    def _map(ex: dict, idx: int | None = None):
        q: str = ex["question"]
        options: Dict[str, str] = ex["options"]
        gold_text: str = ex["answer"]

        gold_letter = None
        for k, v in options.items():
            if (v or "").strip().lower() == (gold_text or "").strip().lower():
                gold_letter = k
                break
        if gold_letter is None:
            return None

        if shuffle_answers and gold_letter in options:
            options, gold_letter, _ = randomize_multiple_choice(
                options=options,
                answer_choice=gold_letter,
                seed=shuffle_seed,
                row_id=ex.get("id", idx),
            )

        return {
            "question": _build_prompt(q, options),
            "answer": gold_letter,
            "info": {
                "answer_text": options.get(gold_letter, gold_text),
                **({"options": options} if shuffle_answers else {}),
            },
        }

    load_from_cache_file = False if shuffle_answers else True
    mapped = ds.map(
        _map,
        with_indices=True,
        remove_columns=ds.column_names,
        load_from_cache_file=load_from_cache_file,
    ).filter(lambda r: r is not None, load_from_cache_file=load_from_cache_file)

    parser = vf.Parser()

    def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(dataset=mapped, eval_dataset=mapped, system_prompt=None, rubric=rubric, parser=parser)
