# MedCalc-Bench

### Overview
- **Environment ID**: `medcalc-bench`
- **Short description**: Evaluate clinical calculator reasoning and numeric/date outputs.
- **Tags**: medical, clinical, single-turn, numeric, date, evaluation

### Dataset
- **Primary dataset**: `ncbi/MedCalc-Bench-v1.2`
- Each example includes `Patient Note`, `Question`, `Calculator ID`, `Ground Truth`, `Lower Bound`, `Upper Bound`.
- Mapped fields in env: `question` (formatted prompt), `calc_id`, `ground_truth`, `lower_bound`, `upper_bound`.

    | Split | Count |
    | ----- | ----- |
    | train | 1100   |
    | test  | 10543  |

### Task
- **Type**: single-turn
- **Prompt**: `_build_prompt(patient_note, question)` instructs `<think>...</think>` and `<answer>...</answer>`.
- **Parser**: `ThinkParser(extract_fn=extract_answer)`; `extract_answer` returns the `<answer>` content.
- **Rubric**: `check_correctness` validates by calculator type:
  - IDs 13, 68: date equality (MM/DD/YYYY)
  - ID 69: tuple `(weeks, days)` equality
  - Integer IDs: integer equality (with rounding as needed)
  - Decimal IDs: numeric value within `[lower_bound, upper_bound]`

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval medcalc-bench
```

Configure model and sampling:

```bash
uv run vf-eval medcalc-bench \
    -m gpt-4.1-mini \
    -n 5 -r 3 -t 1024 -T 0.7 \
    -a '{"use_think": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.


### Environment Arguments

| Arg         | Type | Default | Description |
| ----------- | ---- | ------- | ----------- |
| `use_think` | bool | `False` | Whether to instruct `<think>...</think>` formatting |


### Metrics

| Metric | Meaning |
| ------ | ------- |
| `check_correctness` | (weight 1.0): validates numeric/date/tuple answers per calc ID |

### Adjustments 

Adjusted the prompt to output the step-by-step thinking and final answer with the <think> and <answer> tags instead of responding with a JSON. 
