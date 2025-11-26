import os
import re

import verifiers as vf
from datasets import Dataset, concatenate_datasets
import pandas as pd
import evaluate
from openai import AsyncOpenAI

from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice


# MedExQA specialties
SPECIALTIES = [
    "biomedical_engineer",
    "clinical_laboratory_scientist",
    "clinical_psychologist",
    "occupational_therapist",
    "speech_pathologist",
]

# author prompt directly taken from https://github.com/knowlab/MedExQA/blob/9a5b34af103b0c8ba0c00906e278f6572249fafa/evaluate_pipe_MedExQA.py#L32
def _build_question_str(question: str, options: dict[str, str], use_think: bool = False) -> str:
    """Build user prompt with authors' instruction embedded (as in their script).

    The instruction lives in the user message; the system prompt remains empty in
    normal mode. In think mode, system prompt instructs use of <think> and <answer> tags.
    """
    instruction = (
        "The following is a multiple-choice question. Please choose the most suitable one "
        "among A, B, C and D as the answer to this question. Your answer should be paired "
        "with an explanation why you chose that answer.\n\n"
    )
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{instruction}{question}\n{opts}\nAnswer:"


def _to_vf_format(ds: Dataset, shuffle_answers: bool, shuffle_seed: int | None) -> Dataset:
    """Normalize raw rows into the fields expected by SingleTurnEnv.

    Produces rows of the form:
      - question: string containing authors' instruction, question, and options
      - answer: gold letter (A/B/C/D) - shuffled if shuffle_answers=True
      - info: original fields including exp0/exp1 and specialty, plus shuffled options
    """
    def _format_row(row: dict) -> dict:
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

        # Shuffle options if requested
        if shuffle_answers:
            opts, answer_letter, _ = randomize_multiple_choice(
                options=opts,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=question,  # Use question text for deterministic per-row shuffling
            )

        question_str = _build_question_str(question, opts)

        # Keep original data in info
        info = dict(row)

        # Update info with shuffled values
        if shuffle_answers:
            info["A"] = opts["A"]
            info["B"] = opts["B"]
            info["C"] = opts["C"]
            info["D"] = opts["D"]
            info["answer"] = answer_letter

        return {
            "question": question_str,
            "answer": answer_letter,
            "info": info,
        }

    # Disable cache when shuffling to ensure fresh randomization
    load_from_cache_file = not shuffle_answers
    return ds.map(
        _format_row,
        remove_columns=ds.column_names,
        load_from_cache_file=load_from_cache_file
    ).filter(lambda row: row is not None, load_from_cache_file=load_from_cache_file)


def load_environment(
    use_think: bool = False,
    use_explanations: bool = True,
    mcq_weight: float = 0.5,
    explanation_weight: float = 0.5,
    specialty: list[str] | str | None = None,  # list of short codes or full names; None/"ALL" => all
    explanation_metrics: list[str] | str | None = None,  # None/"all" => average of all four
    # MCQ shuffling
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    # Optional judge settings
    use_judge: bool = False,
    judge_mode: str | None = None,  # "g-eval" | "factscore"
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    use_coverage: bool = False,  # For FactScore: enable coverage calculation (slower but comprehensive)
    **kwargs
) -> vf.Environment:
    """
    Single-turn MedExQA environment using HuggingFace `bluesky333/MedExQA` dataset

    Key behaviors:
      - User prompt embeds the authors' instruction and the options (authors' format).
      - System prompt: empty (normal) or THINK_BOXED (think mode).
      - Specialty selection: accepts list or string; loads requested specialties (None/ALL => all).
      - MCQ accuracy: authors' regex+fuzzy extraction; returns 0 or 100.
      - Explanation score: lexical metrics (ROUGE-L, BLEU, METEOR, BERTScore) averaged 0–100; 0 if answer wrong.
      - Optional judge mode: explanation scored by JudgeRubric (0–100).
    """

    # Load specialties (one or more)
    # Note: MedExQA only has dev and test splits, no train split
    # Load TSV files directly since HF dataset has column name issues

    # Resolve allowed specialties up-front and only load those files
    code_map = {
        "BE": "biomedical_engineer",
        "CLS": "clinical_laboratory_scientist",
        "CP": "clinical_psychologist",
        "OT": "occupational_therapist",
        "SLP": "speech_pathologist",
        "ALL": "all",
    }
    allowed_names: set[str]
    if specialty is None or (isinstance(specialty, str) and (specialty.upper() in ("ALL", ""))):
        allowed_names = set(SPECIALTIES)
    elif isinstance(specialty, str):
        allowed_names = {code_map.get(specialty.upper(), specialty)}
    else:
        tmp = set()
        for s in specialty:
            name = code_map.get((s or "").upper(), s)
            if name and name != "all":
                tmp.add(name)
        allowed_names = tmp if tmp else set(SPECIALTIES)
    macro_active = len(allowed_names) > 1

    # Load all requested specialties
    test_datasets = []
    for sp_name in SPECIALTIES:
        if sp_name not in allowed_names:
            continue
        try:
            url = f"https://huggingface.co/datasets/bluesky333/MedExQA/resolve/main/test/{sp_name}_test.tsv"
            df = pd.read_csv(
                url,
                sep='\t',
                header=None,
                names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"]
            )
            df['specialty'] = sp_name
            ds_part = Dataset.from_pandas(df, preserve_index=False)
            test_datasets.append(ds_part)
        except Exception as e:
            print(f"Warning: Could not load {sp_name}: {e}")
            continue

    # Concatenate and format for verifiers - no training dataset available
    test_combined = concatenate_datasets(test_datasets) if test_datasets else None
    test_ds = _to_vf_format(test_combined, shuffle_answers, shuffle_seed) if test_combined else None

    # Shuffle examples if multiple specialties were selected
    if macro_active and test_ds is not None:
        try:
            test_ds = test_ds.shuffle(seed=int(kwargs.get("seed", 0)))
        except Exception:
            pass

    # Setup system prompt and parser - standardized with medredqa approach
    # - Normal mode: No system prompt, parser returns raw text
    # - Think mode: XML system prompt, XMLParser extracts from <answer> tags
    if use_think:
        # Like medredqa: think in <think> tags, answer+explanation in <answer> tags
        system_prompt = (
            "Think step-by-step inside <think>...</think> tags. "
            "Then, inside <answer>...</answer> tags, provide your final answer choice (A, B, C, or D) "
            "followed by an explanation of why you chose that answer."
        )
        parser = vf.XMLParser(fields=["think", "answer"], answer_field="answer")
    else:
        # Normal mode: no system prompt, parser returns raw text for multiple_choice_accuracy
        system_prompt = ""
        parser = vf.Parser()

    # (shuffling handled above when multiple specialties)

    # Lexical Metrics selection; pass individually or None/'all'/'overall' => average of all four
    base_metrics = ["rougeL", "bleu", "meteor", "bertscore"]
    if explanation_metrics is None:
        selected_metrics = base_metrics
    else:
        if isinstance(explanation_metrics, str) and explanation_metrics.lower() in ("all", "overall"):
            selected_metrics = base_metrics
        elif isinstance(explanation_metrics, list) and any(str(m).lower() in ("all", "overall") for m in explanation_metrics):
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
        return (sum(metric_vals) / len(metric_vals))

    # Note: No per-example macro scaling.

    def _get_completion_text(completion_obj) -> str:
        if isinstance(completion_obj, list) and completion_obj:
            return completion_obj[-1].get("content", "") or ""
        return completion_obj if isinstance(completion_obj, str) else str(completion_obj)

    def answer_accuracy_reward(parser, completion, answer, **kwargs) -> float:
        # Parse answer first (extracts from \boxed{} in think mode, returns raw text in normal mode)
        parsed = parser.parse_answer(completion) or ""
        info = kwargs.get("info", {}) or {}

        # Get answer_text for fallback matching
        options = {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")}
        answer_text = options.get(answer, "")

        is_correct = multiple_choice_accuracy(
            llm_answer=parsed,
            answer_letter=answer,
            answer_text=answer_text,
            accept_answer_text=True,
            strip_tex=False,  # MedExQA doesn't use LaTeX
        )
        return 100.0 if is_correct else 0.0

    def explanation_reward(parser, completion, answer, **kwargs) -> float:
        # Parse answer first (extracts from \boxed{} in think mode, returns raw text in normal mode)
        parsed = parser.parse_answer(completion) or ""
        info = kwargs.get("info", {}) or {}

        # Get answer_text for fallback matching
        options = {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")}
        answer_text = options.get(answer, "")

        # Check if answer is correct using multiple_choice_accuracy
        is_correct = multiple_choice_accuracy(
            llm_answer=parsed,
            answer_letter=answer,
            answer_text=answer_text,
            accept_answer_text=True,
            strip_tex=False,
        )

        if not is_correct:
            return 0.0
        else:
            # For lexical metrics, use the raw completion text (not parsed)
            completion_text = _get_completion_text(completion)
            return compute_expl_score(completion_text, info.get("exp0", ""), info.get("exp1", ""))

    # Optional: Use LLM-as-judge for explanation instead of lexical metrics
    if use_explanations and use_judge:
        api_key = judge_api_key if judge_api_key else os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None
        if judge_mode is None:
            raise ValueError("use_judge=True requires judge_mode to be one of {'g-eval','factscore'}")
        if judge_mode not in ("g-eval", "factscore"):
            raise ValueError("judge_mode must be 'g-eval' or 'factscore'")

        if judge_mode == "g-eval":
            from medexqa.geval_judge.geval_judge import create_geval_judge_rubric
            judge_rubric = create_geval_judge_rubric(parser=parser, judge_client=judge_client, judge_model=judge_model, explanation_weight=explanation_weight)
            # Combine answer accuracy with the judge-based explanation score
            judge_rubric.add_reward_func(answer_accuracy_reward, weight=mcq_weight)
            rubric = judge_rubric
        elif judge_mode == "factscore":
            from medexqa.factscore_judge.atomic_facts_judge import create_factscore_judge_rubric
            judge_rubric = create_factscore_judge_rubric(parser=parser, judge_client=judge_client, judge_model=judge_model, use_coverage=use_coverage, explanation_weight=explanation_weight)
            judge_rubric.add_reward_func(answer_accuracy_reward, weight=mcq_weight)
            rubric = judge_rubric
    else:
        # Lexical metrics for explanations (or MCQ-only if use_explanations=False)
        if use_explanations:
            rubric = vf.Rubric(
                funcs=[answer_accuracy_reward, explanation_reward],
                weights=[mcq_weight, explanation_weight],
                parser=parser
            )
        else:
            # MCQ-only mode
            rubric = vf.Rubric(funcs=[answer_accuracy_reward], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=None,  # No training split available
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )

    return env
