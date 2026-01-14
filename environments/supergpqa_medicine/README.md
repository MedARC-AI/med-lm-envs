# SuperGPQA Medicine

### Overview
- **Environment ID**: `supergpqa_medicine`
- **Short description**: Filtered medicine split from SuperGPQA
- **Tags**: medicine, single-turn, multiple-choice, test, evaluation, supergpqa

### Datasets
- **Primary dataset**: `m-a-p/SuperGPQA` (train split, Medicine discipline only)
- **Source links**: [Paper](https://www.arxiv.org/abs/2502.14739), [GitHub](https://github.com/SuperGPQA/SuperGPQA), [HF Dataset](https://huggingface.co/datasets/m-a-p/SuperGPQA)
- **Split sizes**: 

    | Split (by difficulty)       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `all`  | A-J    | **2755**  |
    | `easy`  | A-J    | **909**  |
    | `middle`  | A-J    | **1629**  |
    | `hard`  | A-J    | **217**  |

### Task
- **Type**: single-turn
- **Parser**: `Parser` with `extract_fn=extract_boxed_answer` for strict letter-in-\boxed{}-format parsing
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval supergpqa_medicine
```

Enable five-shot prompting, shuffle choices, and filter to a field/difficulty:

```bash
uv run vf-eval supergpqa_medicine \
    -m gpt-5 \
    -n -1 -r 3 -t 1024 -T 0.7  \
    -a '{"num_few_shot": 5, "use_think": true, "shuffle_answers": true, "shuffle_seed": 1618, "field": "clinical_medicine", "difficulty": "hard"}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The dataset does have a `validation` split with 3 rows, but these are used as few-shot examples, following the official MMLU-Pro [eval code](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py#L173).
- Setting `use_think` to `True` works best with `num_few_shot` of at least `1`, so that the LLM can learn exactly how it should format its answer.


### Environment Arguments

| Arg        | Type / Choices                                                            | Default | Description |
| ---------- | ------------------------------------------------------------------------- | ------- | ----------- |
| `field`    | `"all"` or one of `basic_medicine`, `clinical_medicine`, `pharmacy`, `public_health_and_preventive_medicine`, `stomatology`, `traditional_chinese_medicine` | `all` | Filter by medical field. |
| `difficulty` | `"all"`, `"easy"`, `"middle"`, `"hard"`                                 | `all` | Filter by question difficulty. |
| `few_shot` | bool                                                                    | `False` | Include fixed five-shot examples in prompts when `True`. |
| `shuffle_answers` | bool                                                              | `False` | Shuffle answer choices per row. |
| `shuffle_seed` | int or `null`                                                        | `1618` | Seed for deterministic shuffling when enabled. |
| `jitter_age` | bool                                                                   | `False` | Add small decimal jitter (~±2 weeks) to age mentions. |

### Metrics

| Metric    | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| `accuracy` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0 |
