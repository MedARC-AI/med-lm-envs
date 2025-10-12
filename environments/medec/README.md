# MEDEC Overview

**Environment ID:** `medec`
**Short Description:**
A benchmark for medical error detection, extraction, and correction in clinical notes, based on the **MEDIQA-CORR 2024** shared task.

**Tags:**
`medical`, `clinical`, `error-detection`, `error-correction`, `single-turn`, `llm-as-judge`, `evaluation`, `metrics`


## Datasets

**Source Links:**
[Paper](https://arxiv.org/html/2412.19260v1), [Original GitHub](https://github.com/abachaa/MEDEC), [HF Dataset](https://huggingface.co/datasets/sauravlmx/MEDEC-MS)

### Split Sizes

| Split         | Count | Description                         |
| ------------- | ----- | ----------------------------------- |
| train_ms      | 2,189 | MS Training Set                     |
| validation_ms | 574   | MS Validation Set with Ground Truth |
| test_ms       | 597   | MS Test Set with Ground Truth       |


## Task

**Type:** `single-turn`
**Parser:** `vf.XMLParser`
**Fields:** `error_flag`, `error_sentence`, `corrected_sentence`


## Rubric Overview

The environment supports two distinct evaluation modes, controlled by the `eval_method` argument:

* **`"judge"` (Default Mode)**
  Uses a **multi-part rubric** where the primary score is derived from a robust *LLM-as-a-Judge* evaluation.

  * Also calculates ROUGE, BERTScore, and BLEURT for reference.
  * These are assigned `weight=0` and **do not affect the primary score**.
  * Recommended for **semantically nuanced evaluation**.

* **`"metrics"` (Replication Mode)**

  * Designed for **direct replication** of the paper's results.
  * Disables the LLM-as-a-Judge.
  * Primary score = **weighted average** of `flag_accuracy` and the paper’s original metrics.


## Quickstart

### 1. Export API Key

The default judge and model under evaluation is **DeepSeek Chat**, which expects the `DEEPSEEK_API_KEY` environment variable.

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 2. Run Evaluation (Default Judge Mode)

Run an evaluation on **10 examples** from the `test_ms` split using the **LLM-as-a-Judge** for scoring.

```bash
uv run vf-eval medec -m deepseek-chat -n 10 -s
```

### 3. Run Evaluation (Paper Replication Mode)

To replicate the paper’s scoring methodology, explicitly set the evaluation mode to `"metrics"`.

```bash
uv run vf-eval medec -m deepseek-chat -a '{"eval_method": "metrics"}' -n 10 -s
```

### 4. Evaluate a Different Model (e.g., Anthropic)

To evaluate an **Anthropic model** while using the default DeepSeek judge:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

uv run vf-eval medec \
  -m "claude-3-5-sonnet-20240620" \
  -b "https://api.anthropic.com/v1" \
  -k ANTHROPIC_API_KEY \
  --header "anthropic-version: 2023-06-01" \
  -n 10 -s
```

## Environment Arguments

| Argument         | Type | Default                         | Description                                                                      |
| ---------------- | ---- | ------------------------------- | -------------------------------------------------------------------------------- |
| `repo_id`        | str  | `"sauravlmx/MEDEC-MS"`          | Hugging Face Hub repository ID for the dataset.                                  |
| `split`          | str  | `"test_ms"`                     | Dataset split to use (`train_ms`, `validation_ms`, `test_ms`).                   |
| `eval_method`    | str  | `"judge"`                       | Evaluation mode (`"judge"` or `"metrics"`).                                      |
| `judge_model`    | str  | `"deepseek-chat"`               | Model used for judge-based scoring (in `"judge"` mode).                          |
| `judge_base_url` | str  | `"https://api.deepseek.com/v1"` | API endpoint for the judge model.                                                |
| `judge_api_key`  | str  | `None`                          | API key for the judge model (defaults to `DEEPSEEK_API_KEY`).                    |
| `device`         | str  | `None`                          | Device to use for metrics (`cpu`, `cuda:0`, etc.). Defaults to GPU if available. |


## Metrics (in Default `"Judge"` Mode)

When `eval_method="judge"`, the following metrics are calculated.
The **primary reward score** is the weighted sum of the first three metrics.

| Metric                   | Weight | Meaning                                                                     |
| ------------------------ | ------ | --------------------------------------------------------------------------- |
| `flag_accuracy`          | 0.2    | 1.0 if predicted `error_flag` matches ground truth; else 0.0.               |
| `extraction_similarity`  | 0.4    | 1.0 if LLM judge deems `error_sentence` semantically equivalent; else 0.0.  |
| `correction_equivalence` | 0.4    | 1.0 if LLM judge deems `corrected_sentence` medically equivalent; else 0.0. |
| `rouge_reward`           | 0      | ROUGE-1 F1 score (for analysis only).                                       |
| `bertscore_reward`       | 0      | BERTScore F1 (for analysis only).                                           |
| `bleurt_reward`          | 0      | BLEURT score (for analysis only).                                           |
| `reward`                 | N/A    | Final weighted sum of non-zero weight metrics (0.0–1.0).                    |