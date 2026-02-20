# careqa

Evaluation environment for the [HPAI-BSC/CareQA](https://huggingface.co/datasets/HPAI-BSC/CareQA) dataset.

## Overview
- **Environment ID**: `careqa`  
- **Short description**: CareQA is a healthcare QA dataset with **multiple-choice** and **open-ended clinical reasoning questions**. This environment supports both modes through the `split` parameter.  
- **Tags**: healthcare, medical QA, clinical reasoning, MCQ, single-turn

## Datasets
- **Primary dataset(s)**:
  - `CareQA_en` – multiple-choice clinical questions with 4 options and correct answer labels
  - `CareQA_en_open` – open-ended clinical questions with reference answers
- **Source links**:
  - [Hugging Face CareQA dataset](https://huggingface.co/datasets/HPAI-BSC/CareQA)

## Task
- **Type**: single-turn
  - MCQ mode: `vf.Parser()` or `vf.ThinkParser()` for extracting boxed answers
  - Open-ended mode: `XMLParser()` for judge responses
- **Rubric overview**:
  - **MCQ mode (`en`)**: `vf.Rubric()` measuring **accuracy** (letter match A–D)
  - **Open-ended mode (`open`)**: LLM-as-judge scoring (single or multi-judge)

## Quickstart

**Multiple-choice evaluation:**
```bash
prime eval run careqa -m "openai/gpt-5-mini" -n 5 -s -a '{"split": "en"}'
```

**Open-ended evaluation:**
```bash
medarc-eval careqa --split open -m "openai/gpt-5-mini" -n 10 -s --judge-model "openai/gpt-5-mini"
```

This environment supports multi-judge via the `medarc-verifiers` `MultiJudge` rubric. When multiple judge models are specified, each scores independently and the final reward is their mean. `judge_model`, `judge_base_url`, and `judge_api_key` are all list-valued and correspond positionally; if only one `judge_base_url` or `judge_api_key` is provided, it is used for all judge models.

**With configured default judges for open-ended mode:**
```bash
medarc-eval careqa --split open -m "openai/gpt-5-mini" -n 10 -s \
  --judge-model "openai/gpt-5-mini" \
  --judge-model "google/gemini-3-flash-preview"
```

**With shuffled answer options (MCQ only):**
```bash
medarc-eval careqa --split en --shuffle-answers --shuffle-seed 1618 -m "openai/gpt-5-mini" -n 10 -s
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | Required | Mode: `en` (multiple-choice) or `open` (open-ended) |
| `system_prompt` | str \| None | `None` | Custom system prompt (uses mode-appropriate default if not specified) |
| `shuffle_answers` | bool | `False` | Randomly shuffle answer options (MCQ only) |
| `shuffle_seed` | int \| None | `1618` | Seed for answer shuffling (MCQ only) |
| `judge_model` | str \| list[str] | `"gpt-4o-mini"` | Model(s) for LLM-as-judge evaluation (open-ended only) |
| `judge_base_url` | str \| list[str] \| None | `None` | Base URL(s) for judge API |
| `judge_api_key` | str \| list[str] \| None | `None` | API key(s) for judge (falls back to `OPENAI_API_KEY` env var) |

## Metrics

### MCQ Mode
| Metric        | Meaning |
|---------------|---------|
| `reward`      | Main scalar reward (weighted sum of rubric criteria) |
| `accuracy`    | Exact match on target MCQ answer (letter A–D) |

### Open-Ended Mode
| Metric        | Meaning |
|---------------|---------|
| `reward`      | Main scalar reward (weighted sum of rubric criteria) |
| `judge_score` | LLM-assigned score evaluating answer quality, correctness, and clinical reasoning |

## Example Usage

```python
import verifiers as vf

# Load MCQ environment
env_mcq = vf.load_environment("careqa", split="en", shuffle_answers=True)

# Load open-ended environment
env_open = vf.load_environment(
    "careqa",
    split="open",
    judge_model=["openai/gpt-5-mini", "google/gemini-3-flash-preview"],
    judge_base_url="https://api.pinference.ai/api/v1",
)
```
