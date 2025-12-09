# M-ARC

### Overview
- **Environment ID**: `m-arc`
- **Short description**: Long-tail medical questions requiring flexible, clinical reasoning. 
- **Tags**: medical, clinical, single-turn, multiple-choice, test, evaluation

### Datasets
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

### Task
- **Type**: single-turn
- **Parser**: `Parser` or `ThinkParser`, with `extract_fn=extract_boxed_answer` for strict letter-in-\boxed{}-format parsing
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval m-arc
```

Configure model and sampling:

```bash
uv run vf-eval m-arc \
    -m gpt-4.1-mini   \
    -n -1 -r 3 -t 1024 -T 0.7  \
    -a '{"use_think": false, "num_few_shot": 1, "shuffle": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The official M-ARC [eval code](https://github.com/dbernardo05/medARC-QA/blob/main/evaluate_from_api.py#L253) loads the entire MMLU-Pro `validation` split to use as few-shot examples. Here, however, we only use rows from the health category, in line with how the official MMLU-Pro [eval code](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L225) filters by category.
- Setting `use_think` to `True` works best with `num_few_shot` of at least `1`, so that the LLM can learn exactly how it should format its answer.


### Environment Arguments

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `num_few_shot`  | int  | `5`    | The number of few-shot examples to use (`-1` for all)                                                                                                                                |
| `use_think`          | bool | `False` | Whether to check for `<think>...</think>` formatting with `ThinkParser`|
| `shuffle`            | bool | `False` | Whether to shuffle answer choices |


### Metrics

| Metric | Meaning |
| ------ | ------- |
| `correct_answer_reward_func` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0|


