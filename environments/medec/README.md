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
**Fields:** `error_id`, `incorrect_sentence`, `correction`


## Rubric Overview

The environment supports two distinct evaluation modes, controlled by the `eval_method` argument:

* **`"judge"` (Default Mode)**
  Uses a **multi-part rubric** where the primary score is derived from a robust *LLM-as-a-Judge* evaluation using a No Free Labels inspired multi-axis judge rubric.

* **`"metrics"` (Replication Mode)**

  * Designed for **direct replication** of the paper's results.
  * Disables the LLM-as-a-Judge and calculates ROUGE, BERTScore, and BLEURT.
  * Primary score = **weighted average** of `flag_accuracy` and the paper’s original metrics.

* **`"both"` (Combined Mode)**
  * Computes both the LLM-as-a-Judge score and the ROUGE, BERTScore, and BLEURT replication metrics.
    * These are assigned `weight=0` and **do not affect the primary score**.
  * Recommended for **semantically nuanced evaluation**.
  * Useful for **comprehensive analysis**.


## Quickstart

### 1. Export API Key

The default judge model is **GPT-4o-mini**, which expects the `OPENAI_API_KEY` environment variable.

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Run Evaluation (Default Judge Mode)

Run an evaluation on **10 examples** from the `test_ms` split using the **LLM-as-a-Judge** for scoring.

```bash
uv run vf-eval medec -m gpt-4o-mini -n 10 -s
```

### 3. Run Evaluation (Paper Replication Mode)

To replicate the paper’s scoring methodology, explicitly set the evaluation mode to `"metrics"`.

```bash
uv run vf-eval medec -m gpt-4o-mini -a '{"eval_method": "metrics"}' -n 10 -s
```

### 4. Evaluate a Different Model (e.g., Anthropic)

To evaluate an **Anthropic model** while using the default OpenAI judge:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

uv run vf-eval medec \
  -m "claude-3-5-sonnet-20240620" \
  -b "https://api.anthropic.com/v1" \
  -k ANTHROPIC_API_KEY \
  --header "anthropic-version: 2023-06-01" \
  -n 10 -s
```

## Environment Arguments

| Argument         | Type | Default         | Description                                                                      |
| ---------------- | ---- | --------------- | -------------------------------------------------------------------------------- |
| `judge_model`    | str  | `"gpt-4o-mini"` | Model used for judge-based scoring (in `"judge"` or `"both"` mode).              |
| `judge_base_url` | str  | `None`          | API endpoint for the judge model (defaults to OpenAI API).                       |
| `judge_api_key`  | str  | `None`          | API key for the judge model (defaults to `OPENAI_API_KEY`).                      |
| `eval_method`    | str  | `"judge"`       | Evaluation mode (`"judge"`, `"metrics"`, or `"both"`).                           |
| `device`         | str  | `None`          | Device to use for metrics (`cpu`, `cuda:0`, etc.). Defaults to GPU if available. |


## Metrics

### Judge Mode (`eval_method="judge"`)

The **primary reward score** is the weighted sum of three metrics:

| Metric             | Weight | Meaning                                                                     |
| ------------------ | ------ | --------------------------------------------------------------------------- |
| `error_flag`       | 1/3    | 1.0 if predicted `error_id` matches ground truth; else 0.0.                 |
| `error_sentence`   | 1/3    | 1.0 if predicted `incorrect_sentence` matches ground truth; else 0.0.       |
| `error_correction` | 1/3    | 1.0 if LLM judge deems `correction` medically equivalent; else 0.0.         |

### Metrics Mode (`eval_method="metrics"`)

For replicating the paper's evaluation methodology:

| Metric           | Weight | Meaning                                                               |
| ---------------- | ------ | --------------------------------------------------------------------- |
| `error_flag`     | 1/3    | 1.0 if predicted `error_id` matches ground truth; else 0.0.           |
| `error_sentence` | 1/3    | 1.0 if predicted `incorrect_sentence` matches ground truth; else 0.0. |
| `rouge_score`    | 1/6    | ROUGE-1 F1 score.                                                     |
| `bertscore`      | 1/6    | BERTScore F1.                                                         |
| `bleurt`         | 1/6    | BLEURT score.                                                         |

### Both Mode (`eval_method="both"`)

Same as Judge mode, plus paper's evaluation metrics with weight 0 (for analysis only):

| Metric        | Weight | Meaning                       |
| ------------- | ------ | ----------------------------- |
| `rouge_score` | 0      | ROUGE-1 F1 score.             |
| `bertscore`   | 0      | BERTScore F1.                 |
| `bleurt`      | 0      | BLEURT score.                 |