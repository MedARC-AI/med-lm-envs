# SuperGPQA Medicine

## Overview
- **Environment ID**: `supergpqa_medicine`
- **Short description**: Filtered medicine split from SuperGPQA
- **Tags**: medicine, single-turn, multiple-choice, test, evaluation, supergpqa

## Datasets
- **Primary dataset**: `m-a-p/SuperGPQA` (train split, Medicine discipline only)
- **Source links**: [Paper](https://www.arxiv.org/abs/2502.14739), [GitHub](https://github.com/SuperGPQA/SuperGPQA), [HF Dataset](https://huggingface.co/datasets/m-a-p/SuperGPQA)
- **Split sizes**: 

    | Split (by difficulty)       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `all`  | A-J    | **2755**  |
    | `easy`  | A-J    | **909**  |
    | `middle`  | A-J    | **1629**  |
    | `hard`  | A-J    | **217**  |

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run supergpqa_medicine -m "openai/gpt-5-mini" -n 5 -s
```

Enable few-shot prompting and filter to a field/difficulty using `medarc-eval`:

```bash
medarc-eval supergpqa_medicine -m "openai/gpt-5-mini" -n -1 --few-shot --field clinical_medicine --difficulty hard
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).
- The dataset does have a `validation` split with 3 rows, but these are used as few-shot examples, following the official MMLU-Pro [eval code](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L173).
- Set `few_shot=true` to include the fixed five-shot examples from the official setup.


## Environment Arguments

| Arg        | Type / Choices                                                            | Default | Description |
| ---------- | ------------------------------------------------------------------------- | ------- | ----------- |
| `field`    | `"all"` or one of `basic_medicine`, `clinical_medicine`, `pharmacy`, `public_health_and_preventive_medicine`, `stomatology`, `traditional_chinese_medicine` | `all` | Filter by medical field. |
| `difficulty` | `"all"`, `"easy"`, `"middle"`, `"hard"`                                 | `all` | Filter by question difficulty. |
| `few_shot` | bool                                                                    | `False` | Include fixed five-shot examples in prompts when `True`. |
| `shuffle_answers` | bool                                                              | `False` | Shuffle answer choices per row. |
| `shuffle_seed` | int or `null`                                                        | `1618` | Seed for deterministic shuffling when enabled. |
| `jitter_age` | bool                                                                   | `False` | Add small decimal jitter (~±2 weeks) to age mentions. |

## Metrics

| Metric    | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| `accuracy` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0 |
