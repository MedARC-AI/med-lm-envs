# medconceptsqa

## Overview
- **Environment ID**: `medconceptsqa`
- **Short description**: MedConcepts QA - an MCQ dataset involving medical codes.
- **Tags**: medical, clinical, single-turn, multiple-choice, classification, test

## Datasets
- **Primary dataset(s)**: `medconceptsqa`
- **Source links**: [Paper](https://www.sciencedirect.com/science/article/pii/S0010482524011740), [Github](https://github.com/nadavlab/MedConceptsQA/tree/master), [HF Dataset](https://huggingface.co/datasets/ofir408/MedConceptsQA)
- **Split sizes**: 60 (dev / few-shot), 820k (test)

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring based on correct answer choice

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medconceptsqa -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medconceptsqa -m "openai/gpt-5-mini" -n 20 --num-few-shot 4
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_few_shot` | int | `0` | Number of few-shot examples to include in the prompt |
| `use_think` | bool | `False` | Whether to use `<think>...</think>` formatting with `ThinkParser` |

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `accuracy` | Exact match on target answer |

## Authors
This environment has been put together by:

Anish Mahishi - ([@macandro96](https://github.com/macandro96))
