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



AUTHOR_SYSTEM_PROMPT = (
    "The following is a multiple-choice question. Please choose the most suitable one "
    "among A, B, C and D as the answer to this question. "
    "Your answer should be paired with an explanation why you chose that answer."
)


def _build_question_str(question: str, options: dict[str, str]) -> str:
    """Format question with answer choices only; instruction is provided via system prompt."""
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{question}\n{opts}\nAnswer:"


def _to_vf_format(ds: Dataset) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': formatted question string with options
      - 'answer': gold letter (A/B/C/D)
      - 'info': keep all original fields including explanations
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
    use_explanations: bool = False,
    mcq_weight: float = 0.5,
    explanation_weight: float = 0.5,
    specialty: str = "all",
    explanation_metrics: list[str] | None = None,
    metrics_aggregation: str = "average",
    macroaverage: bool = False,
    # Optional judge settings
    use_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs
) -> vf.Environment:
    """
    Single-turn MedExQA environment using HuggingFace `bluesky333/MedExQA` dataset

    Each example is normalized to the fields expected by `vf.SingleTurnEnv`:
        {
            "question": "<question + formatted options>",  # string used as the user prompt
            "answer": "<A|B|C|D>",                         # top-level gold letter
            "info": { ...original example fields... }      # full source row including exp0, exp1
        }

    - Loads all 5 medical specialties (biomedical engineering, clinical lab science,
      clinical psychology, occupational therapy, speech language pathology)
    - No training split (dataset does not provide one)
    - Test split used as evaluation data (940 total examples)

    - Parser extracts \\boxed{A|B|C|D} from completions

    - Reward looks for exact match between parsed letter and answer letter
    - Optional: Explanation quality evaluation using LLM-as-judge
    """

    # Load all specialties and concatenate
    # Note: MedExQA only has dev and test splits, no train split
    # Load TSV files directly since HF dataset has column name issues
    test_datasets = []

    for specialty in SPECIALTIES:
        try:
            # Download and load TSV file directly
            url = f"https://huggingface.co/datasets/bluesky333/MedExQA/resolve/main/test/{specialty}_test.tsv"

            # Load TSV with pandas (no headers in file)
            df = pd.read_csv(
                url,
                sep='\t',
                header=None,
                names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"]
            )

            # Add specialty column
            df['specialty'] = specialty

            # Convert to HF dataset
            test_ds = Dataset.from_pandas(df, preserve_index=False)
            test_datasets.append(test_ds)
        except Exception as e:
            print(f"Warning: Could not load {specialty}: {e}")
            continue

    # Concatenate all specialties
    test_combined = concatenate_datasets(test_datasets) if test_datasets else None

    # Format for verifiers - no training dataset available
    test_ds = _to_vf_format(test_combined) if test_combined else None

    # Setup system prompt - use authors' instruction in system; prepend think prompt if requested
    system_prompt = f"{THINK_BOXED_SYSTEM_PROMPT}\n{AUTHOR_SYSTEM_PROMPT}" if use_think else AUTHOR_SYSTEM_PROMPT

    # Parser for extracting \\boxed{} answers
    parser = (
        vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think
        else vf.Parser(extract_fn=extract_boxed_answer)
    )

    def correct_answer_reward_func(parser, completion, answer, **kwargs) -> float:
        """Reward function for MCQ accuracy."""
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    # Optional specialty filter (short codes supported)
    if specialty and test_ds is not None:
        code_map = {
            "BE": "biomedical_engineer",
            "CLS": "clinical_laboratory_scientist",
            "CP": "clinical_psychologist",
            "OT": "occupational_therapist",
            "SLP": "speech_pathologist",
            "ALL": "all",
        }
        spec_upper = (specialty or "all").upper()
        resolved = code_map.get(spec_upper, specialty)
        if resolved != "all":
            test_ds = test_ds.filter(lambda row: (row.get("info") or {}).get("specialty") == resolved)

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

    # Metrics selection; 'all'/'overall' => average of all four
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

    # Precompute specialty counts for macroaverage weighting (if requested)
    spec_counts: dict[str, int] = {}
    total_examples = 0
    if test_ds is not None:
        for row in test_ds:
            info_row = row.get("info") or {}
            spec = info_row.get("specialty") or "unknown"
            spec_counts[spec] = spec_counts.get(spec, 0) + 1
            total_examples += 1
    num_specs = len(spec_counts) if spec_counts else 1

    def _macro_scale(spec: str) -> float:
        if not macroaverage:
            return 1.0
        if spec_counts and total_examples and num_specs:
            n_k = spec_counts.get(spec, 1)
            return float(total_examples) / float(num_specs * n_k)
        return 1.0

    def answer_accuracy_reward(parser, completion, answer, **kwargs) -> float:
        completion_text = completion if isinstance(completion, str) else str(completion)
        info = kwargs.get("info", {}) or {}
        options = {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")}
        gold = (answer or "").strip().upper()
        pred_letter = extract_answer_letter(completion_text, options)
        base = 1.0 if pred_letter == gold else 0.0
        spec = (info.get("specialty") or "unknown")
        return base * _macro_scale(spec)

    def explanation_reward(parser, completion, answer, **kwargs) -> float:
        completion_text = completion if isinstance(completion, str) else str(completion)
        info = kwargs.get("info", {}) or {}
        options = {"A": info.get("A", ""), "B": info.get("B", ""), "C": info.get("C", ""), "D": info.get("D", "")}
        gold = (answer or "").strip().upper()
        pred_letter = extract_answer_letter(completion_text, options)
        if pred_letter != gold:
            base = 0.0
        else:
            base = compute_expl_score(completion_text, info.get("exp0", ""), info.get("exp1", ""))
        spec = (info.get("specialty") or "unknown")
        return base * _macro_scale(spec)

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
            completion_text = completion if isinstance(completion, str) else str(completion)
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
                    "**Correct Answer:** " + str(gold) + "\n\n"
                    "**Reference Explanation 1:**\n" + str(exp0) + "\n\n"
                    "**Reference Explanation 2:**\n" + str(exp1) + "\n\n"
                    "**Model's Response:**\n" + completion_text + "\n\n"
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
                    m = _re.search(r"(\d+\.?\d*)", score_str)
                    s = float(m.group(1)) if m else 0.0
                except Exception:
                    s = 0.0
                base = max(0.0, min(1.0, s)) * 100.0
            spec = (info.get("specialty") or "unknown")
            return base * _macro_scale(spec)

        # Use JudgeRubric with two metrics: answer accuracy (sync), explanation judge (async)
        judge_rubric.add_reward_func(answer_accuracy_reward, weight=0.0)
        judge_rubric.add_reward_func(explanation_judge_reward, weight=0.0)
        rubric = judge_rubric
    else:
        # Keep metrics separate (no combined reward)
        rubric = vf.Rubric(funcs=[answer_accuracy_reward, explanation_reward], weights=[0.0, 0.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=None,  # No training split available
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
