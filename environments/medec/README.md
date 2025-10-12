# **MEDEC Overview**

### **Environment ID:** `medec`

**Short Description:**
A benchmark for medical error detection, extraction, and correction in clinical notes, based on the **MEDIQA-CORR 2024** shared task.

**Tags:**
`medical`, `clinical`, `error-detection`, `error-correction`, `single-turn`, `llm-as-judge`, `evaluation`

---

## **Datasets**

**Source Links:**
[Paper](https://aclanthology.org/2025.findings-acl.1159.pdf), [Original GitHub](https://github.com/abachaa/MEDEC), [HF Dataset](https://huggingface.co/datasets/sauravlmx/MEDEC-MS)

### **Split Sizes**

| Split           | Count | Description                         |
| :-------------- | :---: | :---------------------------------- |
| `train_ms`      | 2,189 | MS Training Set                     |
| `validation_ms` |  574  | MS Validation Set with Ground Truth |
| `test_ms`       |  597  | MS Test Set with Ground Truth       |

---

## **Task**

* **Type:** `single-turn`
* **Parser:** `vf.XMLParser`
  **Fields:** `error_flag`, `error_sentence`, `corrected_sentence`

---

## **Rubric Overview**

A weighted, multi-part rubric evaluating three sub-tasks:

1. **Binary classification** for the presence of an error (`error_flag`).
2. **LLM-as-a-judge** evaluation for the *semantic equivalence* of the extracted error sentence.
3. **LLM-as-a-judge** evaluation for the *medical equivalence* of the generated correction.

---

## **Quickstart**

### **1. Export API Key**

The default judge is **DeepSeek Chat**, which expects the `DEEPSEEK_API_KEY` environment variable.

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### **2. Run Evaluation on Default Split**

Run an evaluation on the default `test_ms` split using **deepseek-chat** as the model under evaluation:

```bash
uv run vf-eval medec -m deepseek-chat -n 10 -s
```

### **3. Evaluate on a Different Split**

Use the `-a` flag to pass environment arguments:

```bash
uv run vf-eval medec -m deepseek-chat -a '{"split": "validation_ms"}' -n 10 -s
```

> **Note:**
> The model specified with `-m` is **the model being evaluated**.
> The model used for scoring (the “judge”) is configured separately via environment arguments (`-a`).
> By default, both are **deepseek-chat**.

---


For eg. to evaluate an **Anthropic model** while using the default **DeepSeek judge**, export both API keys and specify the model, endpoint, and headers:

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

---

## **Environment Arguments**

| Argument         |  Type  |            Default           | Description                                                        |
| :--------------- | :----: | :--------------------------: | :----------------------------------------------------------------- |
| `repo_id`        |  `str` |    `"sauravlmx/MEDEC-MS"`    | The Hugging Face Hub repository ID for the preprocessed dataset.   |
| `split`          |  `str` |          `"test_ms"`         | The dataset split to use (`train_ms`, `validation_ms`, `test_ms`). |
| `judge_model`    |  `str` |       `"deepseek-chat"`      | The model used for judge-based scoring.                            |
| `judge_base_url` |  `str` | `"https://api.deepseek.com"` | The API endpoint for the judge model.                              |
| `judge_api_key`  |  `str` |            `None`            | API key for the judge model.                                       |
| `judge_headers`  | `dict` |            `None`            | Optional headers for the judge model API.                          |

---

## **Metrics**

| Metric                   | Weight | Meaning                                                                                                          |
| :----------------------- | :----: | :--------------------------------------------------------------------------------------------------------------- |
| `flag_accuracy`          |   0.2  | 1.0 if the predicted `error_flag` matches the ground truth, else 0.0.                                            |
| `extraction_similarity`  |   0.4  | 1.0 if the LLM judge deems the extracted `error_sentence` semantically equivalent to the ground truth, else 0.0. |
| `correction_equivalence` |   0.4  | 1.0 if the LLM judge deems the `corrected_sentence` medically equivalent to the ground truth, else 0.0.          |
| `reward`                 |   N/A  | Final weighted sum of the three metrics, resulting in a single score from 0.0 to 1.0.                            |

