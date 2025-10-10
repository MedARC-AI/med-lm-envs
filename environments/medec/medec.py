import os
import csv

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI


def load_environment(
    repo_id: str = "sauravlmx/MEDEC-MS",
    split: str = "test_ms",
    judge_model: str = "deepseek-chat",
    judge_base_url: str = "https://api.deepseek.com",
    judge_api_key: str | None = None,
) -> vf.Environment:
    """
    MEDEC environment for medical error detection and correction.
    Loads the preprocessed MEDEC-MS dataset from the specified Hugging Face Hub repository.
    """
    try:
        print(f"Loading dataset from Hub repo: {repo_id}, split: {split}")
        dataset = load_dataset(repo_id, split=split)
    except Exception as e:
        raise ValueError(f"Could not load split '{split}' from repo '{repo_id}'. Ensure you have access. Error: {e}")

    system_prompt = """You are an expert in clinical note analysis. Your task is to detect and correct potential medical errors in a given clinical text.
    Analyze the text and perform the following three steps:
    1.  Determine if there is an error.
    2.  If an error exists, identify the exact sentence containing the error.
    3.  If an error exists, provide a corrected version of that sentence.

    Respond in the following XML format.

    If an error is found:
    <error_flag>1</error_flag>
    <error_sentence>[The full sentence containing the error]</error_sentence>
    <corrected_sentence>[Your corrected version of the sentence]</corrected_sentence>

    If no error is found:
    <error_flag>0</error_flag>
    """

    parser = vf.XMLParser(
        fields=["error_flag", "error_sentence", "corrected_sentence"]
    )

    api_key = judge_api_key or os.getenv("DEEPSEEK_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key)

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

    def get_final_content(completion) -> str | None:
        """Robustly extracts the content string from the final assistant message."""
        if isinstance(completion, list) and completion:
            last_message = completion[-1]
            if isinstance(last_message, dict):
                content = last_message.get("content")
                if isinstance(content, str):
                    return content
        elif isinstance(completion, str): # Handle cases where completion might be a string
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

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(flag_accuracy, weight=0.2)
    rubric.add_reward_func(extraction_similarity, weight=0.4)
    rubric.add_reward_func(correction_equivalence, weight=0.4)

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

    processed_dataset = dataset.map(
        preprocess_example, remove_columns=dataset.column_names
    )

    return vf.SingleTurnEnv(
        eval_dataset=processed_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

