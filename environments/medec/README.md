# MEDEC

Evaluation environment for the MEDEC dataset.

## Overview
- **Environment ID**: `medec`
- **Short description**: A benchmark for medical error detection, extraction, and correction in clinical notes, based on the MEDIQA-CORR 2024 shared task.
- **Tags**: medical, clinical, error-detection, error-correction, single-turn, llm-as-judge, evaluation, metrics

## Datasets
- **Source links**: [Paper](https://arxiv.org/html/2412.19260v1), [Original GitHub](https://github.com/abachaa/MEDEC), [HF Dataset](https://huggingface.co/datasets/sauravlmx/MEDEC-MS)
- **Split sizes**:

| Split         | Count | Description                         |
| ------------- | ----- | ----------------------------------- |
| train_ms      | 2,189 | MS Training Set                     |
| validation_ms | 574   | MS Validation Set with Ground Truth |
| test_ms       | 597   | MS Test Set with Ground Truth       |

## Task
- **Type**: single-turn
- **Rubric overview**: Supports three evaluation modes via `eval_method`:
  - **`"judge"` (default)**: LLM-as-a-Judge multi-part rubric (No Free Labels inspired multi-axis judge)
  - **`"metrics"` (replication)**: ROUGE, BERTScore, and BLEURT; primary score is weighted average of `flag_accuracy` and paper metrics
  - **`"both"` (combined)**: Judge mode score + paper metrics at weight 0 for analysis

## Quickstart

### 1. Export API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Run Evaluation (Default Judge Mode)

```bash
prime eval run medec -m "openai/gpt-5-mini" -n 5 -s
```

```bash
medarc-eval medec -m "openai/gpt-5-mini" -n 10 -s --judge-model "openai/gpt-5-mini"
```

This environment supports multi-judge via the `medarc-verifiers` `MultiJudge` rubric. When multiple judge models are specified, each scores independently and the final reward is their mean. `judge_model`, `judge_base_url`, and `judge_api_key` are all list-valued and correspond positionally; if only one `judge_base_url` or `judge_api_key` is provided, it is used for all judge models.

```bash
medarc-eval medec -m "openai/gpt-5-mini" -n 10 -s --judge-model "openai/gpt-5-mini" --judge-model "google/gemini-3-flash-preview"
```

### 3. Run Evaluation (Paper Replication Mode)

```bash
medarc-eval medec -m "openai/gpt-5-mini" --eval-method metrics -n 10 -s
```

### 4. Evaluate a Different Model (e.g., Anthropic)

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

medarc-eval medec -m gpt-5-mini -b https://api.anthropic.com/v1 -k ANTHROPIC_API_KEY --header 'anthropic-version: 2023-06-01' -n 10 -s
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

| Metric             | Weight | Meaning                                                                     |
| ------------------ | ------ | --------------------------------------------------------------------------- |
| `error_flag`       | 1/3    | 1.0 if predicted `error_id` matches ground truth; else 0.0.                 |
| `error_sentence`   | 1/3    | 1.0 if predicted `incorrect_sentence` matches ground truth; else 0.0.       |
| `error_correction` | 1/3    | 1.0 if LLM judge deems `correction` medically equivalent; else 0.0.         |

### Metrics Mode (`eval_method="metrics"`)

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
