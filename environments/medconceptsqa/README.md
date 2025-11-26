# medconceptsqa

> Replace the placeholders below, then remove this callout.

### Overview
- **Environment ID**: `medconceptsqa`
- **Short description**: MedConcepts QA - an MCQ dataset involving medical codes.
- **Tags**: medical, clinical, single-turn, multiple-choice, classification, test

### Datasets
- **Primary dataset(s)**: `medconceptsqa`
- **Source links**: [Paper](https://www.sciencedirect.com/science/article/pii/S0010482524011740), [Github](https://github.com/nadavlab/MedConceptsQA/tree/master), [HF Dataset](https://huggingface.co/datasets/ofir408/MedConceptsQA)
- **Split sizes**: 60 (dev / few-shot), 820k (test)

### Task
- **Type**: single-turn
- **Parser**: Uses `extract_boxed_answer` to parse gold letter answer choice. Uses `BOXED_SYSTEM_PROMPT`, `THINK_BOXED_SYSTEM_PROMPT` depending on `use_think`
- **Rubric overview**: Binary scoring based on correct answer choice

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval medconceptsqa
```

Configure model and sampling:

```bash
uv run vf-eval medconceptsqa   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"num_few_shot": 4, "use_think": False}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `accuracy` | Exact match on target answer |

### Authors
This environment has been put together by:

Anish Mahishi - ([@macandro96](https://github.com/macandro96))