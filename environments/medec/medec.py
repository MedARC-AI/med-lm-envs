import os
import zipfile
from enum import Enum

import numpy as np
import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import get_parsed_field
from medarc_verifiers.parsers.xml_parser import XMLParser
from medarc_verifiers.utils import (
    default_judge_api_key,
    download_file,
    judge_sampling_args_and_headers,
    medarc_cache_dir,
)
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State

disable_progress_bar()  # suppress datasets progress indicators


class EvalMethod(str, Enum):
    JUDGE = "judge"
    METRICS = "metrics"
    BOTH = "both"


SYSTEM_PROMPT = """\
Analyze the patient medical narrative and return the following XML:
<error_id>[CORRECT or sentence id]</error_id>
<incorrect_sentence>[incorrect sentence or N/A]</incorrect_sentence>
<correction>[corrected sentence or N/A]</correction>
""".strip()

USER_PROMPT = """The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. The text is either correct or contains one error. The text has one sentence per line. Each line starts with the sentence ID followed by the sentence to check. Check every sentence of the text. If the text is correct return the following output: <error_id>CORRECT</error_id>. If the text has a medical error related to treatment, management, cause, or diagnosis, return the sentence id of the sentence containing the error: <error_id>[sentence id]</error_id>, followed by the incorrect sentence text <incorrect_sentence>[incorrect sentence or N/A]</incorrect_sentence> and a corrected version of the sentence <correction>[corrected sentence or N/A]</correction>. Finding and correcting the error requires medical knowledge and reasoning.
Medical Narrative:
{question}
""".strip()

JUDGE_PROMPT = """\
You are grading an AI assistant's answer to a medical/science exam question where the assistant must fill in a missing sentence in a medical narrative.

Input:
- <context>: The medical narrative with one sentence removed. Each line is numbered with the missing sentence blank.
- <reference_answer>: The original missing sentence and the correct answer.
- <assistant_answer>: The AI's filled in sentence and the response to grade.

Task:
Evaluate the assistant's answer on four Boolean dimensions and output your assessment in the specified format.

Grading Rules:
- Assume the reference answer is correct and reflects the expected exam solution.
- Focus on factual content and meaning, not style, length, or confidence.

Rubric:

1. Semantically Correct (true/false)
- True if the assistant expresses the same core claim(s) as the reference.
- Allow synonyms, paraphrasing, acronyms, and reasonable generalizations that still unambiguously answer the question correctly.
- False if the main concept/mechanism/entity/relationship differs or if the answer is too vague to establish the reference's core claim(s).

2. Matches Details (true/false)
- True if the assistant includes all question-critical details needed to uniquely match the reference answer. Ignore extra illustrative or optional context in the reference.
- False if any required specifics or details from the reference are missing, overgeneralized where precision matters, or incorrect.
- Constraint: If Semantically Correct is false, Matches Details must be false.

3. Substantive Addition (true/false)
- True if the assistant introduces factual claim(s) that could meaningfully alter correctness assessment: tangential or off-topic content, claims or details beyond the question's scope, or alternative explanations/approaches not consistent with the reference.
- False for definitions, brief clarifying context, stylistic elaboration, standard supporting details directly tied to the reference answer, or added specificity that elaborates the same core answer rather than introducing new topics.
- False if the reference answer is incomplete relative to what the question explicitly asks and the assistant provides additional content to fully address the question's stated requirements.

4. Critical Error (true/false)
- True if the assistant states any factual claim that is clearly false relative to the reference and/or standard domain knowledge, or gives unsafe medical guidance.
- False if no clearly incorrect, contradictory, unsafe, or fabricated factual claims are present.
- Note: Missing information alone is not a critical error (it affects Matches Details).
- Note: Critical Error and Substantive Additions are independent; an incorrect added claim may make both true.

<context>
{question}
</context>
<reference_answer>
{answer}
</reference_answer>
<assistant_answer>
{response}
</assistant_answer>

Instructions:
- Briefly compare assistant vs reference for each rubric dimension.
- Output in this exact format:

<analysis>
[Brief dimension-by-dimension analysis]
</analysis>
<semantically_correct>[true/false]</semantically_correct>
<matches_details>[true/false]</matches_details>
<substantive_addition>[true/false]</substantive_addition>
<critical_error>[true/false]</critical_error>
""".strip()


def parse_rubric_scores(ns, name: str, invert: bool = False) -> int:
    raw = get_parsed_field(ns, name, None)
    grade = False
    if raw is None:
        return 0

    if isinstance(raw, bool):
        grade = raw

    if isinstance(raw, str):
        val = raw.strip().lower()
        grade = "true" in val and "false" not in val

    if invert:
        return 0 if grade else 1
    else:
        return 1 if grade else 0


def delete_from_text(text, target_idx):
    target_idx = str(target_idx)
    out_lines = []
    for line in text.splitlines(True):  # keep newline chars
        parts = line.lstrip().split(maxsplit=1)
        if parts and parts[0] == target_idx:
            continue
        out_lines.append(line)
    return "".join(out_lines)


def load_environment(
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    eval_method: str | EvalMethod = EvalMethod.JUDGE,
    device: str | None = None,
) -> vf.Environment:
    """
    MEDEC environment for medical error detection and correction.

    Evaluation Methods:
    - 'judge' (default): The primary score is determined by an LLM-as-a-Judge rubric.
      The paper's original metrics (ROUGE, BERTScore, BLEURT) are also calculated
      and reported, but with a weight of 0, meaning they do not influence the
      main reward score but are available for analysis.
    - 'metrics': The primary score is determined by the paper's original metrics,
      allowing for direct replication of the paper's results.
    - 'judge-only': The primary score is determined solely by the LLM-as-a-Judge rubric,
      without calculating the original metrics.
    """
    if device and "cuda" in device:
        gpu_id = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    eval_method = EvalMethod(eval_method) if isinstance(eval_method, str) else eval_method

    dataset = load_dataset("sauravlmx/MEDEC-MS", split="test_ms")
    train_dataset = load_dataset("sauravlmx/MEDEC-MS", split="train")

    parser = XMLParser(fields=["error_id", "incorrect_sentence", "correction"])
    judge_parser = XMLParser(
        fields=["semantically_correct", "matches_details", "substantive_addition", "critical_error"]
    )

    def error_flag(parser: XMLParser, completion: Messages, info: Info, **kwargs) -> float:
        parsed = parser.parse(completion)
        if parsed is None:
            return 0.0
        try:
            predicted = getattr(parsed, "error_id", None)
            ground_truth = int(info.get("error_id"))
            if ground_truth == -1:
                return "correct" in str(predicted).lower()
            else:
                return 1.0 if int(predicted) == ground_truth else 0.0
        except (ValueError, TypeError):
            return 0.0

    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)

    judge_rubric = vf.JudgeRubric(
        parser=judge_parser,
        parallelize_scoring=True,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    if eval_method in (EvalMethod.METRICS, EvalMethod.BOTH):
        import bert_score
        import bleurt.score as bleurt_score
        from rouge import Rouge

        rouge_model = Rouge()
        bleurt_checkpoint = medarc_cache_dir() / "medec" / "bleurt-20"
        if not bleurt_checkpoint.exists():
            print("Downloading BLEURT-20 checkpoint...")
            url = "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
            zip_path = bleurt_checkpoint.parent / f"{bleurt_checkpoint.name}.zip"
            download_file(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(bleurt_checkpoint.parent)
            zip_path.unlink()
            print("BLEURT-20 checkpoint downloaded and extracted.")
        bleurt_scorer = bleurt_score.BleurtScorer(str(bleurt_checkpoint))
    else:
        rouge_model = None
        bleurt_scorer = None

    def error_sentence(parser: XMLParser, completion: Messages, info: Info, **kwargs) -> float:
        parsed = parser.parse(completion)
        if parsed is None:
            return 0.0

        if error_flag(parser, completion, info, **kwargs) == 0:
            return 0.0  # Incorrect error sentence ID'd, so score is 0

        if int(info.get("error_id")) == -1:
            return 1.0  # No error to extract, so score is 1

        predicted_sentence = getattr(parsed, "incorrect_sentence", "") or ""
        ground_truth_sentence = info.get("error_sentence", "") or ""

        return predicted_sentence.strip() in ground_truth_sentence.strip()

    async def error_correction(parser: XMLParser, completion: Messages, info: Info, state: State, **kwargs) -> float:
        parsed = parser.parse(completion)
        if parsed is None:
            return 0.0

        if error_flag(parser, completion, info, **kwargs) == 0:
            return 0.0  # Incorrect error sentence ID'd, so score is 0

        if int(info.get("error_id")) == -1:
            return 1.0  # No error to extract, so score is 1

        prediction = getattr(parsed, "correction", "") or ""
        ground_truth = info.get("corrected_sentence", "") or ""

        judge_prompt = JUDGE_PROMPT.format(
            question=delete_from_text(info.get("text", ""), info.get("error_id")),
            answer=ground_truth,
            response=prediction,
        )
        try:
            try:
                judge_raw = await judge_rubric.judge(judge_prompt, prediction, ground_truth, state)
                rubric_scores = judge_parser.parse(judge_raw)
            except AttributeError:
                judge_raw = await judge_rubric.judge(judge_prompt, prediction, ground_truth, state)
                rubric_scores = judge_parser.parse(judge_raw)

            semantically_correct = parse_rubric_scores(rubric_scores, "semantically_correct")
            matches_details = parse_rubric_scores(rubric_scores, "matches_details")
            substantive_addition = parse_rubric_scores(rubric_scores, "substantive_addition", invert=True)
            critical_error = parse_rubric_scores(rubric_scores, "critical_error", invert=True)

            score = semantically_correct + matches_details + substantive_addition + critical_error

            info.setdefault("judge_feedback", []).append(
                {
                    "semantically_correct": semantically_correct,
                    "matches_details": matches_details,
                    "substantive_addition": substantive_addition,
                    "critical_error": critical_error,
                    "raw_judge": str(judge_raw),
                }
            )

            return score / 4.0
        except Exception as e:
            print(f"Judge call failed for correction: {e}")
            return 0.0

    def rouge_score(parser: XMLParser, completion: Messages, info: Info, **kwargs) -> float:
        parsed = parser.parse(completion)
        if parsed is None:
            return 0.0

        if error_flag(parser, completion, info, **kwargs) == 0:
            return 0.0  # Incorrect error sentence ID'd, so score is 0

        if int(info.get("error_id")) == -1:
            return 1.0  # No error to extract, so score is 1

        prediction = getattr(parsed, "correction", "") or ""
        ground_truth = info.get("corrected_sentence", "") or ""

        scores = rouge_model.get_scores([prediction], [ground_truth])
        return scores[0]["rouge-1"]["f"]

    def bertscore(parser: XMLParser, completion: Messages, info: Info, **kwargs) -> float:
        parsed = parser.parse(completion)
        if parsed is None:
            return 0.0

        if error_flag(parser, completion, info, **kwargs) == 0:
            return 0.0  # Incorrect error sentence ID'd, so score is 0

        if int(info.get("error_id")) == -1:
            return 1.0  # No error to extract, so score is 1

        prediction = getattr(parsed, "correction", "") or ""
        ground_truth = info.get("corrected_sentence", "") or ""

        _, _, f1 = bert_score.score(
            [prediction], [ground_truth], lang="en", model_type="microsoft/deberta-xlarge-mnli", device=device
        )
        return f1.mean().item()

    def bleurt(parser: XMLParser, completion: Messages, info: Info, **kwargs) -> float:
        parsed = parser.parse(completion)
        if parsed is None:
            return 0.0

        if error_flag(parser, completion, info, **kwargs) == 0:
            return 0.0  # Incorrect error sentence ID'd, so score is 0

        if int(info.get("error_id")) == -1:
            return 1.0  # No error to extract, so score is 1

        prediction = getattr(parsed, "correction", "") or ""
        ground_truth = info.get("corrected_sentence", "") or ""

        scores = bleurt_scorer.score(references=[ground_truth], candidates=[prediction])
        return np.mean(scores)

    final_rubric = vf.Rubric(parser=parser)

    if eval_method == EvalMethod.JUDGE:
        final_rubric.add_reward_func(error_flag, weight=1 / 4)
        final_rubric.add_reward_func(error_sentence, weight=1 / 4)
        final_rubric.add_reward_func(error_correction, weight=1 / 2)
    elif eval_method == EvalMethod.BOTH:
        final_rubric.add_reward_func(error_flag, weight=1 / 4)
        final_rubric.add_reward_func(error_sentence, weight=1 / 4)
        final_rubric.add_reward_func(error_correction, weight=1 / 2)
        final_rubric.add_reward_func(rouge_score, weight=0)
        final_rubric.add_reward_func(bertscore, weight=0)
        final_rubric.add_reward_func(bleurt, weight=0)
    elif eval_method == EvalMethod.METRICS:
        # This mode is for pure replication of the paper's evaluation method.
        final_rubric.add_reward_func(error_flag, weight=1 / 3)
        final_rubric.add_reward_func(error_sentence, weight=1 / 3)
        final_rubric.add_reward_func(rouge_score, weight=1 / 6)
        final_rubric.add_reward_func(bertscore, weight=1 / 6)
        final_rubric.add_reward_func(bleurt, weight=1 / 6)
    else:
        raise ValueError("eval_method must be one of 'judge', 'metrics', or 'both'.")

    def preprocess_example(example: dict) -> dict:
        return {
            "question": USER_PROMPT.format(question=example["Sentences"]),
            "info": {
                "sentences": example["Sentences"],
                "error_id": int(example["Error Sentence ID"]),
                "error_sentence": example["Error Sentence"],
                "corrected_sentence": example["Corrected Sentence"],
            },
            "answer": example["Corrected Sentence"],
        }

    dataset = dataset.map(preprocess_example, remove_columns=dataset.column_names)
    train_dataset = train_dataset.map(preprocess_example, remove_columns=train_dataset.column_names)

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=dataset,
        parser=parser,
        rubric=final_rubric,
        system_prompt=SYSTEM_PROMPT,
    )
