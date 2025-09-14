# medbullets

### Overview
- **Environment ID**: `medbullets`
- **Short description**: USMLE-style multiple-choice questions from Medbullets.
- **Tags**: medical, clinical, single-turn, multiple-choice, USMLE, train, evaluation

### Datasets
- **Primary dataset(s)**: `Medbullets-4` and `Medbullets-5`
- **Source links**: [Paper](https://arxiv.org/pdf/2402.18060), [Github](https://github.com/HanjieChen/ChallengeClinicalQA), [HF Dataset](https://huggingface.co/datasets/mkieffer/Medbullets)
- **Split sizes**: 80/20 train/eval splits 

    | Split       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `op4_train` | {A, B, C, D}    | **246** |
    | `op4_eval`  | {A, B, C, D}    | **62**  |
    | `op5_train` | {A, B, C, D, E} | **246** |
    | `op5_eval`  | {A, B, C, D, E} | **62**  |

    `op5` splits contain the same questions as `op4` splits, but with one additional answer choice to increase difficulty. Note that while the content and answer texts are identical, the correct answer letter may differ between `op4` and `op5`.

### Task
- **Type**: single-turn
- **Parser**: `Parser` or `ThinkParser`, with `extract_fn=extract_boxed_answer` for strict letter-in-\boxed{}-format parsing
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval medbullets
```

Configure model and sampling:

```bash
uv run vf-eval medbullets \
    -m gpt-4.1-mini   \
    -n -1 -r 3 -t 1024 -T 0.7  \
    -a '{"num_train_examples": -1, "num_eval_examples": -1, "num_options": 4, "use_think": true}'

```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `num_train_examples` | int  | `-1`    | Limit the number of training examples (`-1` for all)                                                                                                                            |
| `num_eval_examples`  | int  | `-1`    | Limit the number of eval examples (`-1` for all)                                                                                                                                |
| `num_options`        | int  | `4`     | Number of options: `4` → {A, B, C, D}; `5` → {A, B, C, D, E}                                                |
| `use_think`          | bool | `False` | Whether to check for `<think>...</think>` formatting with `ThinkParser`|

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `correct_answer_reward_func` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0|
| `parser.get_format_reward_func()` | (weight 0.0): optional format adherence (not counted) |


