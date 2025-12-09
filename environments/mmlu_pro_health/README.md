# MMLU-Pro Health

### Overview
- **Environment ID**: `mmlu-pro-health`
- **Short description**: Filtered health split from MMLU-Pro
- **Tags**: medical, clinical, single-turn, multiple-choice, test, evaluation, mmlu

### Datasets
- **Primary dataset(s)**: `MMLU-Pro`
- **Source links**: [Paper](https://arxiv.org/pdf/2406.01574), [Github](https://github.com/TIGER-AI-Lab/MMLU-Pro), [HF Dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- **Split sizes**: 

    | Split       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `test`  | A-J    | **553**  |

### Task
- **Type**: single-turn
- **Parser**: `Parser` or `ThinkParser`, with `extract_fn=extract_boxed_answer` for strict letter-in-\boxed{}-format parsing
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval mmlu-pro-health
```

Configure model and sampling (overriding some environment arguments):

```bash
uv run vf-eval mmlu-pro-health \
    -m gpt-4.1-mini   \
    -n -1 -r 3 -t 1024 -T 0.7  \
    -a '{"use_think": false, "num_few_shot": 3, "shuffle_answers": true, "shuffle_seed": 1618}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The dataset does have a `validation` split with 3 rows, but these are used as few-shot examples, following the official MMLU-Pro [eval code](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L173).
- Setting `use_think` to `True` works best with `num_few_shot` of at least `1`, so that the LLM can learn exactly how it should format its answer.


### Environment Arguments

| Arg               | Type           | Default | Description                                                                                                      |
| ----------------- | -------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `num_few_shot`    | int            | `5`     | The number of few-shot examples to use (`-1` for all).                                                           |
| `use_think`       | bool           | `False` | Use `<think>...</think>` formatting with `ThinkParser`.         |
| `shuffle_answers` | bool           | `False` | Shuffle answers choices.                                                                               |
| `shuffle_seed`    | int or `null`  | `1618`  | Deterministic seed for choice shuffling when `shuffle_answers` is `True` (`null` for non-deterministic).         |
| `jitter_age`      | bool           | `False` | Add a small decimal jitter (~±2 weeks) to ages in the question text (M-ARC style). |


### Metrics

| Metric    | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| `accuracy` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0 |


