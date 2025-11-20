# ACI Bench Overview

### Overview
- **Environment ID**: `acibench`
- **Short description**: A benchmark for generating structured clinical notes from doctor-patient dialogues, utilizing context-aware prompts and supporting both LLM-as-a-Judge and standard NLP metric evaluations.
- **Tags**: `medical`, `clinical`, `summarization`, `note-generation`, `single-turn`, `evaluation`, `llm-as-judge`

### Datasets
- **Primary dataset(s)**: `harsh-c137/aci-bench-medarc-eval` on Hugging Face. This is the cleaned, and de-duplicated version of the ACI-BENCH corpus, containing 387 unique encounters.
- **Source links**: [Paper](https://doi.org/10.1038/s41597-023-02487-3), [HF Dataset](https://huggingface.co/datasets/harsh-c137/aci-bench-medarc-eval)
- **Split sizes**: The script loads the entire dataset and performs a reproducible 20/80 split in-memory, resulting in:
    - **Train**: 77
    - **Validation**: 310

### Task
- **Type**: `single-turn`
- **Parser**: `vf.Parser` using a custom function to extract the text content from the model's completion object.
- **Context-Aware Dynamic Prompting**: This is the core feature of the environment. Before each evaluation, the script inspects the metadata of the transcript (`transcript_listener` and `transcript_writer`) and the section headers of the ground-truth `note`. It then generates a highly-tailored prompt that instructs the model on how to handle the specific type of conversation (e.g., natural dialogue, dictation, or interaction with a virtual assistant) and provides the exact section headers to use.
- **Rubric Overview**: The environment supports two distinct evaluation modes controlled by the `eval_method` argument:
    - **`judge` (Default)**: Uses a powerful LLM to grade the generated note on Accuracy, Completeness, and Clarity. Standard metrics (ROUGE/BERTScore) are also calculated for reference but do not affect the primary reward.
    - **`metrics`**: Uses a weighted sum of ROUGE, BERTScore, and BLEURT as the primary reward.

### Quickstart

**1. Default Evaluation (LLM-as-a-Judge)**
Runs the evaluation using `gpt-4o` as the judge (requires `OPENAI_API_KEY`).

```bash
export OPENAI_API_KEY="your-openai-api-key"
uv run vf-eval acibench -m gpt-4-turbo -n 5 -s
```

**2. Customizing the Judge, while using Mistral for main task of note generation**
Use a different model (e.g., `gpt-4o-mini`) or provider for the judge.

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
uv run vf-eval acibench \
  -m mistral-small-latest \
  -b https://api.mistral.ai/v1 \
  -k MISTRAL_API_KEY \
  --env-args '{"judge_model": "gpt-4o-mini"}' \
  -n 5 -s
```

**3. Metrics-Only Mode (Legacy)**
Disable the judge and use only ROUGE, BERTScore, and BLEURT for scoring.

```bash
uv run vf-eval acibench \
  -m gpt-4-turbo \
  --env-args '{"eval_method": "metrics"}' \
  -n 5 -s
```

**4. Few-Shot Evaluation**
Add `num_few_shot` to any command to include in-context examples.

```bash
uv run vf-eval acibench \
  -m gpt-4-turbo \
  --env-args '{"num_few_shot": 1}' \
  -n 5 -s
```

### Environment Arguments

| Arg              | Type         | Default   | Description                                                                 |
| ---------------- | ------------ | --------- | --------------------------------------------------------------------------- |
| `eval_method`    | `str`        | `"judge"` | The evaluation mode: `"judge"` (LLM scoring) or `"metrics"` (NLP metrics).  |
| `judge_model`    | `str`        | `"gpt-4o"`| The model identifier to use for the judge (e.g., `gpt-4o`, `gpt-4o-mini`).  |
| `judge_base_url` | `str | None` | `None`    | Custom API base URL for the judge model.                                    |
| `judge_api_key`  | `str | None` | `None`    | API key for the judge. Defaults to `JUDGE_API_KEY` or `OPENAI_API_KEY`.     |
| `num_few_shot`   | `int`        | `0`       | Number of in-context examples to provide in the prompt (0 for zero-shot).   |
| `device`         | `str | None` | `None`    | Device for metric computation (e.g., `"cuda:0"`). Defaults to CPU.          |
| `system_prompt`  | `str | None` | `None`    | An optional custom system prompt to override the default.                   |

### Metrics

**Primary Metrics (in `judge` mode):**

| Metric | Meaning |
| ------ | ------- |
| `judge_reward` | The primary score (0.0 - 1.0). Calculated as the average of the normalized Accuracy, Completeness, and Clarity scores. |
| `accuracy` | (1-5 Scale) Does the note correctly capture main medical issues and avoid hallucinations? |
| `completeness` | (1-5 Scale) Does the note include all important information from the dialogue? |
| `clarity` | (1-5 Scale) Is the note well-structured and easy to read? |

**Secondary Metrics (Primary in `metrics` mode):**

| Metric | Meaning |
| ------ | ------- |
| `rouge` | Measures lexical overlap (exact words/phrases) against the reference note. |
| `bertscore` | Measures semantic similarity using contextual embeddings. |
| `bleurt` | A learned metric trained to predict human judgments of text quality. |

### Author
This environment was developed by **[Harsh Deshpande](https://www.linkedin.com/in/harsh-deshpande-v1/)**. Contributions include the dataset curation and cleaning for Medarc's evaluation task, and creation of the verifiers environment script (`acibench.py`).