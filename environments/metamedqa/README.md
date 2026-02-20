# MetaMedQA

Evaluation environment for the MetaMedQA dataset.

## Overview
- **Environment ID**: `metamedqa`
- **Short description**: Single-turn medical multiple-choice QA drawn from multiple medical exam sources
- **Tags**: medical, single-turn, multiple-choice, eval

## Datasets
- **Primary dataset(s)**: MetaMedQA
- **Source links**: [maximegmd/MetaMedQA](https://huggingface.co/datasets/maximegmd/MetaMedQA)
- **Split sizes**: Uses provided test split

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring (1.0 / 0.0) based on correct letter or answer text match

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run metamedqa -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval metamedqa -m "openai/gpt-5-mini" -n 20 --shuffle-answers --shuffle-seed 1618
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Dataset split to use |
| `shuffle_answers` | bool | `False` | Whether to shuffle answer choices |
| `shuffle_seed` | int \| None | `1618` | Seed for deterministic answer shuffling |

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `accuracy` | (weight 1.0): 1.0 if parsed letter matches the gold letter, else 0.0 |

## Authors
This environment has been put together by:

Aymane Ouraq - ([@aymaneo](https://github.com/aymaneo))
