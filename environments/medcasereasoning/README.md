# medcasereasoning

### Overview
- **Environment ID**: `medcasereasoning`
- **Short description**: MedCaseReasoning dataset from Stanford (Wu et al. 2025)
- **Tags**: 

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: https://huggingface.co/datasets/zou-lab/MedCaseReasoning
- **Split sizes**: 13.1k (train) / 500 (val) / 897 (test)

### Task
- **Type**: single-turn
- **Parser**: JudgeRubric
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval medcasereasoning
```

Configure model and sampling:

```bash
uv run vf-eval medcasereasoning   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7
```

