import os
import zipfile

import bert_score
import bleurt.score
import numpy as np
import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from openai import AsyncOpenAI
from rouge import Rouge

from medarc_verifiers.utils import download_file, medarc_cache_dir

disable_progress_bar()  # suppress datasets progress indicators


def extract_xml_from_string(text: str) -> str:
    """Extracts XML content from a string."""
    import re

    match = re.search(r"<error_flag>.*</corrected_sentence>|<error_flag>0</error_flag>", text, re.DOTALL)
    if match:
        return match.group(0)
    return ""


def load_environment(
    repo_id: str = "sauravlmx/MEDEC-MS",
    test_split: str = "test_ms",
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    num_few_shot: int = 0,
    use_think: bool = False,
    eval_method: str = "judge",  # "judge" (default), "metrics, or "judge-only"
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

    try:
        print(f"Loading dataset from Hub repo: {repo_id}, split: {test_split}")
        dataset = load_dataset(repo_id, split=test_split)
        train_dataset = load_dataset(repo_id, split="train_ms")
    except Exception as e:
        raise ValueError(f"Could not load split '{test_split}' from repo '{repo_id}'. Error: {e}")

    zero_shot_prompt = """The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. The text is either correct or contains one error. The text has one sentence per line. Each line starts with the sentence ID, followed by a pipe character then the sentence to check. Check every sentence of the text. If the text is correct return the following output: <error_flag>0</error_flag>. If the text has a medical error related to treatment, management, cause, or diagnosis, return the sentence id of the sentence containing the error, followed by a space, and then a corrected version of the sentence in the following XML format:
<error_flag>1</error_flag>
<error_sentence>[The full sentence containing the error]</error_sentence>
<corrected_sentence>[Your corrected version of the sentence]</corrected_sentence>
Finding and correcting the error requires medical knowledge and reasoning."""

    few_shot_examples = []
    if num_few_shot > 0 and train_dataset:
        few_shot_dataset = train_dataset.shuffle(seed=42).select(range(num_few_shot))
        for example in few_shot_dataset:
            question = example["Text"]
            if int(example["Error Flag"]) == 1:
                answer = f"""<error_flag>1</error_flag>
<error_sentence>{example["Error Sentence"]}</error_sentence>
<corrected_sentence>{example["Corrected Sentence"]}</corrected_sentence>"""
            else:
                answer = "<error_flag>0</error_flag>"
            few_shot_examples.append({"role": "user", "content": question})
            few_shot_examples.append({"role": "assistant", "content": answer})

    if use_think:
        system_prompt = (
            "Think step-by-step inside <think>...</think> tags. Then, provide your final answer in the specified XML format."
            + "\n\n"
            + zero_shot_prompt
        )
        parser = vf.ThinkParser(extract_fn=extract_xml_from_string)
    else:
        system_prompt = zero_shot_prompt
        parser = vf.XMLParser(fields=["error_flag", "error_sentence", "corrected_sentence"])

    def get_final_content(completion) -> str | None:
        if isinstance(completion, list) and completion:
            last_message = completion[-1]
            if isinstance(last_message, dict):
                content = last_message.get("content")
                if isinstance(content, str):
                    return content
        elif isinstance(completion, str):
            return completion
        return None

    def flag_accuracy(parser, completion, info, **kwargs) -> float:
        final_content = get_final_content(completion)
        if final_content is None:
            return 0.0
        parsed = parser.parse(final_content)
        predicted_flag = getattr(parsed, "error_flag", None)
        ground_truth_flag = info.get("error_flag")
        if predicted_flag is None:
            return 0.0
        try:
            return 1.0 if int(predicted_flag) == ground_truth_flag else 0.0
        except (ValueError, TypeError):
            return 0.0

    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=judge_api_key)
    rouge = Rouge()
    if eval_method is not None and eval_method != "judge-only":
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
        bleurt_scorer = bleurt.score.BleurtScorer(str(bleurt_checkpoint))
    else:
        bleurt_scorer = None

    EXTRACTION_JUDGE_TEMPLATE = """Your job is to evaluate if a predicted sentence is semantically equivalent to a ground truth sentence from a clinical note.

Consider these guidelines:
- Minor wording differences are acceptable if the core meaning is preserved.
- The predicted sentence can be a substring or superset of the ground truth as long as it correctly isolates the error.

Context Clinical Note:
{context}

Ground Truth Error Sentence:
{answer}

Predicted Error Sentence:
{response}

Is the predicted sentence semantically equivalent to the ground truth sentence?
Respond with either "EQUIVALENT" or "NOT_EQUIVALENT".
""".strip()

    CORRECTION_JUDGE_TEMPLATE = """Your job is to evaluate if a predicted correction for a medical error is medically equivalent to a ground truth correction.

Context Clinical Note:
{context}

Sentence with Error:
{error_sentence}

Ground Truth Corrected Sentence:
{answer}

Predicted Corrected Sentence:
{response}

Is the predicted correction medically equivalent to the ground truth correction?
Respond with either "EQUIVALENT" or "NOT_EQUIVALENT".
""".strip()

    async def extraction_similarity(parser, completion, info, **kwargs) -> float:
        final_content = get_final_content(completion)
        if final_content is None:
            return 0.0
        parsed = parser.parse(final_content)
        predicted_sentence = getattr(parsed, "error_sentence", "") or ""
        ground_truth_sentence = info.get("error_sentence", "") or ""
        ground_truth_flag = info.get("error_flag")

        if ground_truth_flag == 0:
            return 1.0 if not predicted_sentence else 0.0
        if not predicted_sentence:
            return 0.0

        judge_prompt = EXTRACTION_JUDGE_TEMPLATE.format(
            context=info.get("text", ""),
            answer=ground_truth_sentence,
            response=predicted_sentence,
        )
        try:
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
            )
            judge_response = response.choices[0].message.content or ""
            return 1.0 if "EQUIVALENT" in judge_response.upper() else 0.0
        except Exception as e:
            print(f"Judge call failed for extraction: {e}")
            return 0.0

    async def correction_equivalence(parser, completion, info, **kwargs) -> float:
        final_content = get_final_content(completion)
        if final_content is None:
            return 0.0
        parsed = parser.parse(final_content)
        predicted_correction = getattr(parsed, "corrected_sentence", "") or ""
        ground_truth_correction = info.get("corrected_sentence", "") or ""
        ground_truth_flag = info.get("error_flag")

        if ground_truth_flag == 0:
            return 1.0 if not predicted_correction else 0.0
        if not predicted_correction:
            return 0.0

        judge_prompt = CORRECTION_JUDGE_TEMPLATE.format(
            context=info.get("text", ""),
            error_sentence=info.get("error_sentence", ""),
            answer=ground_truth_correction,
            response=predicted_correction,
        )
        try:
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
            )
            judge_response = response.choices[0].message.content or ""
            return 1.0 if "EQUIVALENT" in judge_response.upper() else 0.0
        except Exception as e:
            print(f"Judge call failed for correction: {e}")
            return 0.0

    def get_nlg_eval_data(reference, prediction):
        if reference == "NA" or prediction == "NA":
            return None, None
        return [reference], [prediction]

    def rouge_reward(parser, completion, info, **kwargs) -> float:
        final_content = get_final_content(completion)
        if final_content is None:
            return 0.0
        parsed = parser.parse(final_content)
        predicted_correction = getattr(parsed, "corrected_sentence", "NA") or "NA"
        ground_truth_correction = info.get("corrected_sentence", "NA") or "NA"
        refs, preds = get_nlg_eval_data(ground_truth_correction, predicted_correction)
        if not refs:
            return 1.0 if ground_truth_correction == predicted_correction else 0.0
        scores = rouge.get_scores(preds, refs)
        return scores[0]["rouge-1"]["f"]

    def bertscore_reward(parser, completion, info, **kwargs) -> float:
        final_content = get_final_content(completion)
        if final_content is None:
            return 0.0
        parsed = parser.parse(final_content)
        predicted_correction = getattr(parsed, "corrected_sentence", "NA") or "NA"
        ground_truth_correction = info.get("corrected_sentence", "NA") or "NA"
        refs, preds = get_nlg_eval_data(ground_truth_correction, predicted_correction)
        if not refs:
            return 1.0 if ground_truth_correction == predicted_correction else 0.0
        _, _, f1 = bert_score.score(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli", device=device)
        return f1.mean().item()

    def bleurt_reward(parser, completion, info, **kwargs) -> float:
        final_content = get_final_content(completion)
        if final_content is None:
            return 0.0
        parsed = parser.parse(final_content)
        predicted_correction = getattr(parsed, "corrected_sentence", "NA") or "NA"
        ground_truth_correction = info.get("corrected_sentence", "NA") or "NA"
        refs, preds = get_nlg_eval_data(ground_truth_correction, predicted_correction)
        if not refs:
            return 1.0 if ground_truth_correction == predicted_correction else 0.0
        scores = bleurt_scorer.score(references=refs, candidates=preds)
        return np.mean(scores)

    final_rubric = vf.Rubric(parser=parser)
    if eval_method == "judge":
        final_rubric.add_reward_func(flag_accuracy, weight=0.2)
        final_rubric.add_reward_func(extraction_similarity, weight=0.4)
        final_rubric.add_reward_func(correction_equivalence, weight=0.4)
        final_rubric.add_reward_func(rouge_reward, weight=0)
        final_rubric.add_reward_func(bertscore_reward, weight=0)
        final_rubric.add_reward_func(bleurt_reward, weight=0)
    elif eval_method == "metrics":
        # This mode is for pure replication of the paper's evaluation method.
        final_rubric.add_reward_func(flag_accuracy, weight=0.2)
        final_rubric.add_reward_func(rouge_reward, weight=0.8 / 3)
        final_rubric.add_reward_func(bertscore_reward, weight=0.8 / 3)
        final_rubric.add_reward_func(bleurt_reward, weight=0.8 / 3)
    elif eval_method == "judge-only":
        final_rubric.add_reward_func(flag_accuracy, weight=0.2)
        final_rubric.add_reward_func(extraction_similarity, weight=0.4)
        final_rubric.add_reward_func(correction_equivalence, weight=0.4)
    else:
        raise ValueError("eval_method must be one of 'judge', 'metrics', or 'judge-only'")

    def preprocess_example(example: dict) -> dict:
        return {
            "question": example["Text"],
            "info": {
                "text": example["Text"],
                "error_flag": int(example["Error Flag"]),
                "error_sentence": example["Error Sentence"],
                "corrected_sentence": example["Corrected Sentence"],
            },
            "answer": "",
        }

    processed_dataset = dataset.map(preprocess_example, remove_columns=dataset.column_names)

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=processed_dataset,
        system_prompt=system_prompt,
        few_shot=few_shot_examples,
        parser=parser,
        rubric=final_rubric,
    )
