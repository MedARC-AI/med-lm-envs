"""
ACI-BENCH Environment

This script defines an ACI-BENCH evaluation environment compatible with the Verifiers framework.
It sets up the dataset for clinical note generation from doctor-patient dialogues and
configures a rubric with metrics inspired by the original paper: ROUGE, BERTScore, and BLEURT.

This version uses dynamic prompting, extracting the section headers from each ground-truth
note to provide a precise structure for the model to follow.
"""

import os
import zipfile
from pathlib import Path

import bert_score
import bleurt.score
import numpy as np
import verifiers as vf
from datasets import load_dataset
from rouge import Rouge


def medarc_cache_dir() -> Path:
    """Returns the path to the MedARC cache directory."""
    default_cache_path = Path.home() / ".cache" / "medarc"
    cache_dir = Path(os.getenv("MEDARC_CACHE_DIR", default_cache_path))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_file(url: str, destination: Path):
    """Downloads a file from a URL to a destination, showing a progress bar."""
    import requests
    from tqdm import tqdm

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

# --- Core Environment Definition ---

def _get_text_from_completion(completion: any) -> str:
    """Extracts a text string from a model completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last_item = completion[-1]
        if isinstance(last_item, dict):
            return str(last_item.get("content", ""))
        return str(last_item)
    return str(completion)

# Extract section headers from a clinical note.
def _extract_headers_from_note(note_text: str) -> list[str]:
    """Finds all lines in a note that are in ALL CAPS to use as section headers."""
    headers = []
    for line in note_text.splitlines():
        stripped_line = line.strip()
        # A line is considered a header if it's not empty and all its letters are uppercase.
        if stripped_line and stripped_line.isupper():
            headers.append(stripped_line)
    return headers

# Build a dynamic prompt using the extracted headers.
def _build_dynamic_prompt(dialogue: str, headers: list[str]) -> str:
    """Builds a prompt that instructs the LLM to use a specific set of section headers."""
    header_list_str = ", ".join(headers)
    prompt = (
        f"You will be given a dialogue from a doctor-patient encounter. "
        f"Your task is to summarize the dialogue into a clinical note using the following section headers, in this exact order: {header_list_str}.\n\n"
        f"The conversation is:\n\n{dialogue}"
    )
    return prompt


def load_environment(
    system_prompt: str | None = None,
    device: str | None = None,
    num_few_shot: int = 0,
) -> vf.Environment:
    """
    Loads the ACI-BENCH environment for clinical note generation.
    """
    if device and "cuda" in device:
        gpu_id = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Load and split the dataset
    try:
        dataset = load_dataset("harsh-c137/aci-bench-medarc-eval", split="test")
        split_datasets = dataset.train_test_split(test_size=0.2, seed=42)
        train_ds = split_datasets["train"]
        val_ds = split_datasets["test"]
    except Exception as e:
        raise ConnectionError(f"Could not load dataset 'harsh-c137/aci-bench-medarc-eval' from Hugging Face Hub. Error: {e}")

    # --- Prepare Few-Shot Examples (with dynamic prompts) ---
    few_shot_examples = []
    if num_few_shot > 0:
        if len(train_ds) < num_few_shot:
            raise ValueError(
                f"Requested {num_few_shot} few-shot examples, but the training set only has {len(train_ds)} examples."
            )
        
        few_shot_ds = train_ds.shuffle(seed=42).select(range(num_few_shot))
        
        for example in few_shot_ds:
            # Generate a dynamic prompt for each few-shot example
            headers = _extract_headers_from_note(example["note"])
            user_content = _build_dynamic_prompt(example["dialogue"], headers)
            assistant_content = example["note"]
            
            few_shot_examples.append({"role": "user", "content": user_content})
            few_shot_examples.append({"role": "assistant", "content": assistant_content})


    # Preprocess the evaluation examples (with dynamic prompts)
    def _map_example(example: dict) -> dict:
        # Generate a dynamic prompt for each evaluation example
        headers = _extract_headers_from_note(example["note"])
        return {
            "question": _build_dynamic_prompt(example["dialogue"], headers),
            "answer": example["note"],
        }

    train_mapped = train_ds.map(_map_example, remove_columns=train_ds.column_names)
    val_mapped = val_ds.map(_map_example, remove_columns=val_ds.column_names)

    # --- Rubric and Evaluation Metric Setup ---
    parser = vf.Parser(extract_fn=_get_text_from_completion)
    rouge_scorer = Rouge()

    bleurt_checkpoint = medarc_cache_dir() / "acibench" / "bleurt-20"
    if not bleurt_checkpoint.exists():
        print("Downloading BLEURT-20 checkpoint (one-time setup)...")
        url = "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
        zip_path = bleurt_checkpoint.parent / f"{bleurt_checkpoint.name}.zip"
        bleurt_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(bleurt_checkpoint.parent)
        zip_path.unlink()
        print("BLEURT-20 checkpoint downloaded and extracted.")
    bleurt_scorer = bleurt.score.BleurtScorer(str(bleurt_checkpoint))

    def rouge_reward(completion: any, answer: str, **kwargs) -> float:
        clean_completion = _get_text_from_completion(completion)
        if not clean_completion.strip() or not answer.strip():
            return 0.0
        try:
            scores = rouge_scorer.get_scores(clean_completion, answer, avg=True)
            avg_f1 = np.mean([
                scores["rouge-1"]["f"],
                scores["rouge-2"]["f"],
                scores["rouge-l"]["f"],
            ])
            return float(avg_f1)
        except Exception:
            return 0.0

    def bertscore_reward(completion: any, answer: str, **kwargs) -> float:
        clean_completion = _get_text_from_completion(completion)
        if not clean_completion.strip() or not answer.strip():
            return 0.0
        _, _, f1 = bert_score.score(
            [clean_completion], [answer], lang="en", model_type="microsoft/deberta-xlarge-mnli", device=device
        )
        return f1.mean().item()

    def bleurt_reward(completion: any, answer: str, **kwargs) -> float:
        clean_completion = _get_text_from_completion(completion)
        if not clean_completion.strip() or not answer.strip():
            return 0.0
        scores = bleurt_scorer.score(references=[answer], candidates=[clean_completion])
        return np.mean(scores)

    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            rouge_reward,
            bertscore_reward,
            bleurt_reward,
        ],
        weights=[1.0, 1.0, 1.0],
        names=["rouge", "bertscore", "bleurt"],
    )

    final_system_prompt = system_prompt or (
        "You are a helpful medical assistant who generates clinical notes from doctor-patient conversations."
    )

    env = vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=val_mapped,
        few_shot=few_shot_examples,
        system_prompt=final_system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env