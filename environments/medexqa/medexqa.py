from enum import Enum
from pathlib import Path

import evaluate
import pandas as pd
import verifiers as vf
from datasets import Dataset, concatenate_datasets
from medarc_verifiers.parsers import XMLParser
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils import (
    default_judge_api_key,
    download_file,
    judge_sampling_args_and_headers,
    medarc_cache_dir,
)
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from openai import AsyncOpenAI
from verifiers.types import Info, State


class Specialty(str, Enum):
    BIOMEDICAL_ENGINEER = "biomedical_engineer"
    CLINICAL_LABORATORY_SCIENTIST = "clinical_laboratory_scientist"
    CLINICAL_PSYCHOLOGIST = "clinical_psychologist"
    OCCUPATIONAL_THERAPIST = "occupational_therapist"
    SPEECH_PATHOLOGIST = "speech_pathologist"
    ALL = "all"


SYSTEM_PROMPT = "Provide your explanation inside <explanation>...</explanation> tags, then give your final answer inside <answer>...</answer> tags."


JUDGE_TEMPLATE = """\
You are grading an AI assistant's reasoning for a medical multiple-choice question. The assistant selected the correct answer.

Input:
- <question>: The question and answer options
- <answer>: The correct answer choice
- <reference_reasoning>: Two correct reasoning traces
- <assistant_reasoning>: The AI's reasoning to grade

Task: Determine if the assistant's reasoning is EQUIVALENT or INEQUIVALENT by comparing it to the reference reasoning traces and output your grade in <grade>...</grade> tags.

Grading Rules:
- Assume the reference reasoning traces are correct.
- Focus on logical content and decision criteria, not style, length, or confidence.
- Do not solve the question yourself; only compare the assistant's reasoning to the references.

EQUIVALENT if the assistant's reasoning is semantically aligned with at least one reference trace, including:
- Paraphrasing or rephrasing with equivalent logic
- Synonyms, acronyms (expanded or abbreviated), or different medical terminology with the same meaning
- Omitting minor details while preserving central reasoning and decision criteria
- Additional supporting details that don't contradict the reference
- Different ordering of steps that reaches the same logical conclusion

INEQUIVALENT if any of these apply:
- Contradicts both reference reasoning traces on key logical steps
- Uses incompatible logic or decision criteria compared to both references
- Relies on a different main reason that conflicts with both references
- Contains clearly incorrect medical reasoning

Be strict: clear contradictions or incompatible logic with both references = INEQUIVALENT.

<question>
{question}
</question>

<answer>{answer}</answer>

<reference_reasoning>
{reference_1}
</reference_reasoning>

<reference_reasoning>
{reference_2}
</reference_reasoning>

<assistant_reasoning>
{assistant_reasoning}
</assistant_reasoning>

Briefly explain whether the assistant's reasoning aligns with or conflicts with the references. Then output your grade as:

<grade>[Equivalent or Inequivalent]</grade>
""".strip()


# author prompt directly taken from https://github.com/knowlab/MedExQA/blob/9a5b34af103b0c8ba0c00906e278f6572249fafa/evaluate_pipe_MedExQA.py#L32
def _build_question_str(question: str, options: dict[str, str]) -> str:
    """Build user prompt with authors' instruction embedded (as in their script).

    The instruction lives in the user message; the system prompt remains empty in
    normal mode, and only adds THINK_BOXED in think-mode.
    """
    instruction = (
        "The following is a multiple-choice question. Please choose the most suitable one "
        "among A, B, C and D as the answer to this question. Your answer should be paired "
        "with an explanation why you chose that answer.\n\n"
    )
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{instruction}{question}\n{opts}\nAnswer:"


def _resolve_cache_dir(cache_dir: Path | str | None = None) -> Path:
    resolved = medarc_cache_dir(cache_dir) / "medexqa"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _to_vf_format(ds: Dataset, shuffle_answers: bool = False, shuffle_seed: int | None = 1618) -> Dataset:
    """Normalize raw rows into the fields expected by SingleTurnEnv.

    Produces rows of the form:
      - question: string containing authors' instruction, question, and options
      - answer: gold letter (A/B/C/D)
      - info: original fields including exp0/exp1 and specialty
    """

    def _format_row(row: dict, idx: int | None = None) -> dict:
        question = row.get("question", "") or ""

        # Build options dict from A, B, C, D columns
        opts = {
            "A": row.get("A", ""),
            "B": row.get("B", ""),
            "C": row.get("C", ""),
            "D": row.get("D", ""),
        }

        # Get answer letter
        answer_letter = (row.get("answer") or "").strip().upper()
        if answer_letter not in ("A", "B", "C", "D"):
            return None

        if shuffle_answers and answer_letter in opts:
            opts, answer_letter, _ = randomize_multiple_choice(
                options=opts,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=row.get("question") or idx,
            )

        answer_text = opts.get(answer_letter, "")

        question_str = _build_question_str(question, opts)

        # Keep original data in info
        info = dict(row)
        info["answer_text"] = answer_text
        info["answer"] = answer_letter
        info["question"] = question
        if shuffle_answers:
            info["options"] = opts

        return {
            "question": question_str,
            "answer": answer_letter,
            "info": info,
        }

    return ds.map(
        _format_row, with_indices=True, remove_columns=ds.column_names, load_from_cache_file=not shuffle_answers
    ).filter(lambda row: row is not None)


def load_environment(
    use_explanations: bool = True,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    cache_dir: Path | str | None = None,
    specialty: list[str] | str | None = None,  # list of short codes or full names; None/"ALL" => all
    explanation_metrics: list[str] | str | None = None,  # None/"all" => average of all four
    # Optional judge settings
    use_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs,
) -> vf.Environment:
    """
    Single-turn MedExQA environment using HuggingFace `bluesky333/MedExQA` dataset

    Key behaviors:
      - User prompt embeds the authors' instruction and the options (authors' format).
      - System prompt: asks for reasoning in <rationale> tags and final answer in <answer> tags.
      - Specialty selection: accepts list or string; loads requested specialties (None/ALL => all).
      - Optional answer shuffling for robustness (keeps options in info when enabled).
      - Unified scoring: MCQ must be correct or the score is 0; if MCQ is correct but explanation fails, score is 0.5; if both pass, score is 1.0.
      - Explanation check: lexical metrics (ROUGE-L, BLEU, METEOR, BERTScore) or JudgeRubric (LLM-as-a-judge).
    """

    # Load specialties (one or more)
    # Note: MedExQA only has dev and test splits, no train split
    # Load TSV files directly since HF dataset has column name issues

    cache_path = _resolve_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Resolve allowed specialties up-front and only load those files
    if specialty is None:
        specialty = Specialty.ALL
    else:
        specialty = Specialty(specialty) if isinstance(specialty, str) else specialty

    if specialty == Specialty.ALL:
        selected_specialties = [specialty for specialty in Specialty if specialty != Specialty.ALL]
    else:
        selected_specialties = [specialty]

    # Load all requested specialties (with caching)
    test_datasets = []
    for sp in selected_specialties:
        sp_name = sp.value
        try:
            url = f"https://huggingface.co/datasets/bluesky333/MedExQA/resolve/main/test/{sp_name}_test.tsv"
            dest_path = cache_path / f"{sp_name}_test.tsv"
            download_file(url=url, dest=dest_path, verify=False)
            df = pd.read_csv(
                dest_path, sep="\t", header=None, names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"]
            )
            df["specialty"] = sp_name
            ds_part = Dataset.from_pandas(df, preserve_index=False)
            test_datasets.append(ds_part)
        except Exception as e:
            print(f"Warning: Could not load {sp_name}: {e}")
            continue

    # Concatenate and format for verifiers - no training dataset available
    test_combined = concatenate_datasets(test_datasets) if test_datasets else None
    test_ds = (
        _to_vf_format(test_combined, shuffle_answers=shuffle_answers, shuffle_seed=shuffle_seed)
        if test_combined
        else None
    )

    # Shuffle examples if multiple specialties were selected
    if len(selected_specialties) > 1 and test_ds is not None:
        try:
            test_ds = test_ds.shuffle(seed=int(kwargs.get("seed", 0)))
        except Exception:
            pass

    parser = XMLParser(fields=["explanation", "answer"], answer_field="answer")

    # Lexical Metrics selection; pass individually or None/'all'/'overall' => average of all four
    base_metrics = ["rougeL", "bleu", "meteor", "bertscore"]
    if explanation_metrics is None:
        selected_metrics = base_metrics
    else:
        if isinstance(explanation_metrics, str) and explanation_metrics.lower() in ("all", "overall"):
            selected_metrics = base_metrics
        elif isinstance(explanation_metrics, list) and any(
            str(m).lower() in ("all", "overall") for m in explanation_metrics
        ):
            selected_metrics = base_metrics
        else:
            selected_metrics = explanation_metrics

    def compute_metric_score(metric_name: str, prediction: str, refs: list[str]) -> float:
        try:
            name = metric_name.lower()
            if name in ("rouge", "rougel"):
                rouge = evaluate.load("rouge")
                res = rouge.compute(predictions=[prediction], references=[refs])
                return float(res.get("rougeL", 0.0)) * 100.0
            if name == "bleu":
                bleu = evaluate.load("bleu")
                res = bleu.compute(predictions=[prediction], references=[refs])
                sc = float(res.get("bleu", 0.0))
                return sc * 100.0 if sc <= 1.0 else sc
            if name == "meteor":
                meteor = evaluate.load("meteor")
                res = meteor.compute(predictions=[prediction], references=[refs])
                sc = float(res.get("meteor", 0.0))
                return sc * 100.0 if sc <= 1.0 else sc
            if name == "bertscore":
                bscore = evaluate.load("bertscore")
                res = bscore.compute(
                    predictions=[prediction],
                    references=[refs],
                    model_type="allenai/scibert_scivocab_uncased",
                    lang="en",
                    rescale_with_baseline=False,
                )
                f1_list = res.get("f1", [])
                return (float(f1_list[0]) * 100.0) if f1_list else 0.0
            return 0.0
        except Exception:
            return 0.0

    def compute_expl_score(pred: str, exp0: str, exp1: str) -> float:
        refs = [exp0 or "", exp1 or ""]
        metric_vals = [compute_metric_score(m, pred, refs) for m in selected_metrics]
        metric_vals = [v for v in metric_vals if v is not None]
        if not metric_vals:
            return 0.0
        # always average across selected metrics
        return sum(metric_vals) / len(metric_vals)

    # Note: No per-example macro scaling.

    def _is_correct(parser, completion, answer: str, info: dict | None = None) -> bool:
        completion_text = completion or ""
        parsed = parser.parse_answer(completion) or completion_text
        answer_text = (info or {}).get("answer_text", "")
        return multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)

    def combined_reward(parser, completion, answer, **kwargs) -> float:
        """Gate explanation scoring on MCQ correctness."""
        info = kwargs.get("info", {}) or {}
        if not _is_correct(parser, completion, answer, info):
            return 0.0
        if not use_explanations:
            return 1.0
        completion_text = completion or ""
        expl_score = compute_expl_score(completion_text, info.get("exp0", ""), info.get("exp1", ""))
        explanation_passes = expl_score > 0.0
        return 1.0 if explanation_passes else 0.5

    # Optional: Use LLM-as-judge for explanation instead of lexical metrics
    if use_explanations and use_judge:
        api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
        sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
        judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)
        judge_rubric = vf.JudgeRubric(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt="{question}",
            sampling_args=sampling_args,
        )
        judge_parser = XMLParser(fields=["grade"], answer_field="grade")

        async def combined_judge_reward(judge, prompt, completion, answer, state: State, info: Info) -> float:
            answer = answer.strip().upper()
            answer_text = info.get("answer_text", "")
            parsed = parser.parse(completion, last=True)
            model_answer = getattr(parsed, "answer", None)
            model_rational = getattr(parsed, "explanation", None)

            is_correct = multiple_choice_accuracy(
                llm_answer=model_answer, answer_letter=answer, answer_text=answer_text
            )

            if not is_correct:
                return 0.0

            options = info.get(
                "options",
                {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")},
            )

            question = info.get("question", "")
            opts_str = "\n".join(f"{k}. {options.get(k, '')}" for k in ["A", "B", "C", "D"])
            formatted_question = f"{question}\n{opts_str}"

            judge_prompt = JUDGE_TEMPLATE.format(
                question=formatted_question,
                answer=answer,
                reference_1=info.get("exp0", ""),
                reference_2=info.get("exp1", ""),
                assistant_reasoning=model_rational,
            )

            try:
                judge_response = await judge_rubric.judge(judge_prompt, "", "", state)
                grade = judge_parser.parse_answer(judge_response)
            except AttributeError:
                judge_response = await judge_rubric.judge(judge_prompt, "", "", state)
                grade = judge_parser.parse_answer(judge_response)

            try:
                grade = grade.strip().lower()
                explanation_passes = "equivalent" in grade and "inequivalent" not in grade

                info.setdefault("judge_feedback", []).append(
                    {
                        "grade": grade,
                        "raw_judge": str(judge_response),
                    }
                )
            except Exception:
                explanation_passes = False
            return 1.0 if explanation_passes else 0.5

        judge_rubric.add_reward_func(combined_judge_reward, weight=1.0)
        rubric = judge_rubric
    else:
        rubric = vf.Rubric(funcs=[combined_reward], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=None,  # No training split available
        eval_dataset=test_ds,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )

    return env
