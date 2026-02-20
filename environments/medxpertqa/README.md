# medxpertqa

## Overview
- **Environment ID**: `medxpertqa`
- **Short description**: MedXpertQA is a highly challenging and comprehensive benchmark designed to evaluate expert-level medical knowledge and advanced reasoning capabilities. We only use the text subset for now.
- **Tags**: mcq

## Datasets
- **Primary dataset(s)**: TsinghuaC3I/MedXpertQA
- **Source links**: [HuggingFace](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)
- **Split sizes**: test subset - 2.45k rows

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring (1.0 / 0.0) based on correct letter or answer text match

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medxpertqa -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medxpertqa -m "openai/gpt-5-mini" -n 20 --answer-format boxed
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `question_type` | str | `"all"` | Question subset to evaluate (e.g., all, text-only subset variants supported by the environment). |
| `use_think` | bool | `False` | Whether to expect reasoning in `<think>...</think>` with boxed answers. |
| `shuffle_answers` | bool | `False` | Whether to shuffle answer options per question. |
| `shuffle_seed` | int \| None | `1618` | Seed for deterministic answer shuffling. |
| `answer_format` | str | `"xml"` | Output format parser to use (`xml` or `boxed`). |

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |
