# medbullets

## Overview
- **Environment ID**: `medbullets`
- **Short description**: USMLE-style multiple-choice questions from Medbullets.
- **Tags**: medical, clinical, single-turn, multiple-choice, USMLE, train, evaluation

## Datasets
- **Primary dataset(s)**: `Medbullets-4` and `Medbullets-5`
- **Source links**: [Paper](https://arxiv.org/pdf/2402.18060), [Github](https://github.com/HanjieChen/ChallengeClinicalQA), [HF Dataset](https://huggingface.co/datasets/mkieffer/Medbullets)
- **Split sizes**:

    | Split       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `op4_test` | {A, B, C, D}    | **308** |
    | `op5_test` | {A, B, C, D, E} | **308** |

    `op5_test` contains the same content as `op4_test`, but with one additional answer choice to increase difficulty. Note that while the content is the same, the letter choice corresponding to the correct answer is sometimes different between these splits.


## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medbullets -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medbullets -m "openai/gpt-5-mini" -n -1 --num-options 5 --shuffle-answers --shuffle-seed 1618 --answer-format boxed
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).

## Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `num_options`        | int  | `4`     | Number of options: `4` → {A, B, C, D}; `5` → {A, B, C, D, E}                                                |
| `use_think`          | bool | `False` | Whether to check for `<think>...</think>` formatting with `ThinkParser`|
| `shuffle_answers`    | bool | `False` | Whether to shuffle answer choices |
| `shuffle_seed`       | int \| None | `1618` | Seed for deterministic answer shuffling |
| `answer_format`      | str | `"boxed"` | Output parser format: `"boxed"` or `"xml"` |


## Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `correct_answer_reward_func` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0|
| `parser.get_format_reward_func()` | (weight 0.0): optional format adherence (not counted) |
