import os
import re

import verifiers as vf
from datasets import Dataset, concatenate_datasets
from verifiers.utils.data_utils import THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer
import pandas as pd
import evaluate
from thefuzz import process
from openai import AsyncOpenAI


# MedExQA specialties
SPECIALTIES = [
    "biomedical_engineer",
    "clinical_laboratory_scientist",
    "clinical_psychologist",
    "occupational_therapist",
    "speech_pathologist",
]

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


def _to_vf_format(ds: Dataset) -> Dataset:
    """Normalize raw rows into the fields expected by SingleTurnEnv.

    Produces rows of the form:
      - question: string containing authors' instruction, question, and options
      - answer: gold letter (A/B/C/D)
      - info: original fields including exp0/exp1 and specialty
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

        question_str = _build_question_str(question, opts)

        # Keep original data in info
        info = dict(row)

        return {
            "question": question_str,
            "answer": answer_letter,
            "info": info,
        }

    return ds.map(_format_row, remove_columns=ds.column_names).filter(lambda row: row is not None)


def load_environment(
    use_think: bool = False,
    use_explanations: bool = True,
    mcq_weight: float = 0.5,
    explanation_weight: float = 0.5,
    specialty: list[str] | str | None = None,  # list of short codes or full names; None/"ALL" => all
    explanation_metrics: list[str] | str | None = None,  # None/"all" => average of all four
    # Optional judge settings
    use_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
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
    test_ds = _to_vf_format(test_combined) if test_combined else None

    # Shuffle examples if multiple specialties were selected
    if macro_active and test_ds is not None:
        try:
            test_ds = test_ds.shuffle(seed=int(kwargs.get("seed", 0)))
        except Exception:
            pass

    # Setup system prompt - empty for normal; use think-boxed for think mode
    system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else ""

    # Parser for extracting \\boxed{} answers
    parser = (
        vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think
        else vf.Parser(extract_fn=extract_boxed_answer)
    )

    def correct_answer_reward_func(parser, completion, answer, **kwargs) -> float:
        """Reward function for MCQ accuracy."""
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    # (shuffling handled above when multiple specialties)

    # Helpers (authors' answer extraction logic)
    def process_before_extraction(gen: str, choice_dict: dict[str, str]) -> str:
        for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1] or ""), reverse=True):
            pattern = re.compile(re.escape((val or "").rstrip(".")), re.IGNORECASE)
            gen = pattern.sub(key, gen)
        return gen

    def extract_choice(gen: str, choice_list: list[str]) -> str:
        res = re.search(r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b", gen)
        if res is None:
            res = re.search(r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b", gen)
        if res is None:
            res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)
        if res is None:
            res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)
        if res is None:
            best = process.extractOne(gen, choice_list)
            choices = ["A", "B", "C", "D"]
            return choices[choice_list.index(best[0])] if best else ""
        return res.group(1)

    def extract_answer_letter(completion_text: str, options: dict[str, str]) -> str:
        gen = process_before_extraction(completion_text or "", options)
        pred = extract_choice(gen, [options.get(c, "") for c in ["A", "B", "C", "D"]])
        return (pred or "").upper()

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
        completion_text = _get_completion_text(completion)
        info = kwargs.get("info", {}) or {}
        options = {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")}
        gold = (answer or "").strip().upper()
        pred_letter = extract_answer_letter(completion_text, options)
        base = 100.0 if pred_letter == gold else 0.0
        return base

    def explanation_reward(parser, completion, answer, **kwargs) -> float:
        completion_text = _get_completion_text(completion)
        info = kwargs.get("info", {}) or {}
        options = {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")}
        gold = (answer or "").strip().upper()
        pred_letter = extract_answer_letter(completion_text, options)
        if pred_letter != gold:
            base = 0.0
        else:
            base = compute_expl_score(completion_text, info.get("exp0", ""), info.get("exp1", ""))
        return base

    # Optional: Use LLM-as-judge for explanation instead of lexical metrics
    if use_explanations and use_judge:
        api_key = judge_api_key if judge_api_key else os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None
        judge_rubric = vf.JudgeRubric(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt="{question}",
        )

        async def explanation_judge_reward(judge, prompt, completion, answer, state, **kwargs) -> float:
            completion_text = _get_completion_text(completion)
            info = kwargs.get("info", {}) or {}
            options = {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")}
            gold = (answer or "").strip().upper()
            pred_letter = extract_answer_letter(completion_text, options)
            if pred_letter != gold:
                base = 0.0
            else:
                # Build judge prompt
                question = info.get("question", "")
                opts_str = "\n".join(f"{k}. {options.get(k, '')}" for k in ["A","B","C","D"]) 
                formatted_question = f"{question}\n{opts_str}"
                exp0 = info.get("exp0", "")
                exp1 = info.get("exp1", "")
                full_prompt = (
                    "You are evaluating the quality of a medical explanation.\n\n"
                    "**Question:**\n" + formatted_question + "\n\n"
                    "**Correct Answer:** " + str(answer) + "\n\n"
                    "**Reference Explanation 1:**\n" + str(exp0) + "\n\n"
                    "**Reference Explanation 2:**\n" + str(exp1) + "\n\n"
                    "**Model's Response:**\n" + completion_text + "\n\n"
                    "Evaluate whether the model's explanation is medically accurate, relevant, and demonstrates understanding of the medical concepts. The explanation should justify why the answer is correct.\n\n"
                    "Compare the model's explanation quality to the reference explanations. Consider:\n"
                    "- Medical accuracy\n"
                    "- Relevance to the question\n"
                    "- Clarity and completeness\n"
                    "- Proper use of medical concepts\n\n"
                    "Respond with a score from 0.0 to 1.0:\n"
                    "- 1.0 = Excellent (as good as or better than references)\n"
                    "- 0.75 = Good (mostly correct with minor issues)\n"
                    "- 0.5 = Acceptable (partially correct)\n"
                    "- 0.25 = Poor (significant errors)\n"
                    "- 0.0 = Wrong or irrelevant\n\n"
                    "Respond with ONLY a number between 0.0 and 1.0."
                )
                judge_response = await judge_rubric.judge(
                    [{"role": "user", "content": full_prompt}],
                    "",
                    "",
                    state,
                    **kwargs,
                )
                try:
                    score_str = str(judge_response).strip()
                    import re as _re
                    number_match = _re.search(r"(\d+\.?\d*)", score_str)
                    explanation_score = float(number_match.group(1)) if number_match else 0.0
                except Exception:
                    explanation_score = 0.0
                base = max(0.0, min(1.0, explanation_score)) * 100.0
            return base

        # Use JudgeRubric with two metrics: answer accuracy (sync), explanation judge (async)
        judge_rubric.add_reward_func(answer_accuracy_reward, weight=mcq_weight)
        judge_rubric.add_reward_func(explanation_judge_reward, weight=explanation_weight)
        rubric = judge_rubric
    else:
        # Keep metrics separate (and a combine drewad with tunable weights)
        rubric = vf.Rubric(funcs=[answer_accuracy_reward, explanation_reward], weights=[mcq_weight, explanation_weight], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=None,  # No training split available
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )

    return env
