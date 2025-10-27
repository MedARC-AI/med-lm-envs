"""
ACI-BENCH Environment

This script defines an ACI-BENCH evaluation environment compatible with the Verifiers framework.
It sets up the dataset for clinical note generation from doctor-patient dialogues and
configures a rubric with metrics from the original paper: ROUGE, BERTScore, BLEURT, and MEDCON.

Reference to the original paper:
@article{Yim_2023,
    author = {Yim, Wen-wai and Fu, Yujuan and Ben Abacha, Asma and Snider, Neal and Lin, Thomas and Yetisgen, Meliha},
    title = {Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
    journal = {Scientific Data},
    year = {2023},
    month = {sep},
    day = {06},
    volume = {10},
    number = {1},
    doi = {10.1038/s41597-023-02487-3},
    url = {https://doi.org/10.1038/s41597-023-02487-3}
}
"""

import os
import warnings
import zipfile
from pathlib import Path

import bert_score
import bleurt.score
import numpy as np
import verifiers as vf
from datasets import load_dataset
from rouge import Rouge

# --- Utility Functions (adapted from medec.py for self-containment) ---

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


def load_environment(
    system_prompt: str | None = None,
    device: str | None = None,
) -> vf.Environment:
    """
    Loads the ACI-BENCH environment for clinical note generation.

    This function prepares the ACI-BENCH dataset and configures a rubric with
    ROUGE, BERTScore, and BLEURT metrics. A placeholder for MEDCON is included,
    which requires a local QuickUMLS installation to be fully functional.
    """
    # Set CUDA device if specified
    if device and "cuda" in device:
        gpu_id = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Load and split the dataset
    try:
        # CORRECTED LINE: Load the available 'test' split first.
        dataset = load_dataset("harsh-c137/aci-bench-basic", split="test")
        
        # Now, create a reproducible 80/20 train-validation split from the loaded data.
        split_datasets = dataset.train_test_split(test_size=0.2, seed=42)
        train_ds = split_datasets["train"]
        val_ds = split_datasets["test"]
    except Exception as e:
        raise ConnectionError(f"Could not load dataset 'harsh-c137/aci-bench-basic' from Hugging Face Hub. Error: {e}")

    # Define the prompt structure based on the paper's experiments
    def _build_prompt(dialogue: str) -> str:
        prompt = (
            "summarize the conversation to generate a clinical note with four sections: "
            "HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN. "
            f"The conversation is:\n\n{dialogue}"
        )
        return prompt

    # Preprocess examples
    def _map_example(example: dict) -> dict:
        return {
            "question": _build_prompt(example["dialogue"]),
            "answer": example["note"],
        }

    train_mapped = train_ds.map(_map_example, remove_columns=train_ds.column_names)
    val_mapped = val_ds.map(_map_example, remove_columns=val_ds.column_names)

    # --- Rubric and Evaluation Metric Setup ---
    parser = vf.Parser(extract_fn=_get_text_from_completion)
    rouge_scorer = Rouge()

    # Setup BLEURT, including checkpoint download
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

    # Define reward functions for each metric
    def rouge_reward(completion: str, answer: str, **kwargs) -> float:
        """Calculates the average of ROUGE-1, ROUGE-2, and ROUGE-L F1-scores."""
        if not completion.strip() or not answer.strip():
            return 0.0
        try:
            scores = rouge_scorer.get_scores(completion, answer, avg=True)
            avg_f1 = np.mean([
                scores["rouge-1"]["f"],
                scores["rouge-2"]["f"],
                scores["rouge-l"]["f"],
            ])
            return float(avg_f1)
        except Exception:
            return 0.0

    def bertscore_reward(completion: str, answer: str, **kwargs) -> float:
        """Calculates the BERTScore F1."""
        if not completion.strip() or not answer.strip():
            return 0.0
        _, _, f1 = bert_score.score(
            [completion], [answer], lang="en", model_type="microsoft/deberta-xlarge-mnli", device=device
        )
        return f1.mean().item()

    def bleurt_reward(completion: str, answer: str, **kwargs) -> float:
        """Calculates the BLEURT score."""
        if not completion.strip() or not answer.strip():
            return 0.0
        scores = bleurt_scorer.score(references=[answer], candidates=[completion])
        return np.mean(scores)

    def medcon_reward(completion: str, answer: str, **kwargs) -> float:
        """
        Placeholder for MEDCON.
        A full implementation requires a local QuickUMLS installation.
        See: https://github.com/Georgetown-IR-Lab/QuickUMLS
        """
        warnings.warn(
            "MEDCON metric is not fully implemented and will return 0.0. "
            "A local QuickUMLS installation is required for a complete implementation."
        )
        return 0.0

    # Assemble the final rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            rouge_reward,
            bertscore_reward,
            bleurt_reward,
            medcon_reward,
        ],
        weights=[1.0, 1.0, 1.0, 1.0], # Equal weights to ensure all are calculated
        names=["rouge", "bertscore", "bleurt", "medcon"],
    )

    # Set a default system prompt if not provided
    final_system_prompt = system_prompt or (
        "You are a helpful assistant who generates a clinical note from a doctor-patient conversation."
    )

    # Create and return the environment
    env = vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=val_mapped,
        system_prompt=final_system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env