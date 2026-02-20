# M-ARC

## Overview
- **Environment ID**: `m-arc`
- **Short description**: Long-tail medical questions requiring flexible, clinical reasoning. 
- **Tags**: medical, clinical, single-turn, multiple-choice, test, evaluation

## Datasets
- **Primary dataset**: `M-ARC`
- **Source links**: [Paper](https://arxiv.org/pdf/2502.04381), [Github](https://github.com/dbernardo05/medARC-QA), [HF Dataset](https://huggingface.co/datasets/mkieffer/M-ARC)
- **Split sizes**: 

    | Split       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `test`  | A-G    | **100**  |

- **Few-shot dataset**: `MMLU-Pro-Health`
- **Source links**: [Paper](https://arxiv.org/pdf/2406.01574), [Github](https://github.com/TIGER-AI-Lab/MMLU-Pro), [HF Dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- **Split sizes**: 

    | Split       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `validation`  | A-J    | **3**  |

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run m-arc -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval m-arc -m "openai/gpt-5-mini" -n -1 -s --num-few-shot 1 --shuffle-answers --shuffle-seed 1618
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).
- The official M-ARC [eval code](https://github.com/dbernardo05/medARC-QA/blob/main/evaluate_from_api.py#L253) loads the entire MMLU-Pro `validation` split to use as few-shot examples. Here, however, we only use rows from the health category, in line with how the official MMLU-Pro [eval code](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L225) filters by category.
- Setting `use_think` to `True` works best with `num_few_shot` of at least `1`, so that the LLM can learn exactly how it should format its answer.


## Environment Arguments

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `num_few_shot`  | int  | `5`    | The number of few-shot examples to use (`-1` for all)                                                                                                                                |
| `use_think`          | bool | `False` | Whether to check for `<think>...</think>` formatting with `ThinkParser`|
| `shuffle_answers`    | bool | `False` | Whether to shuffle answer choices |
| `shuffle_seed`       | int \| None | `1618` | Seed for deterministic answer shuffling |


## Metrics

| Metric | Meaning |
| ------ | ------- |
| `correct_answer_reward_func` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0|
