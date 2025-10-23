# medxpertqa


### Overview
- **Environment ID**: `medxpertqa`
- **Short description**: MedXpertQA is a highly challenging and comprehensive benchmark designed to evaluate expert-level medical knowledge and advanced reasoning capabilities. We only use the text subset for now.
- **Tags**: mcq

### Datasets
- **Primary dataset(s)**: TsinghuaC3I/MedXpertQA
- **Source links**: [HuggingFace](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)
- **Split sizes**: test subset - 2.45k rows

### Task
- **Type**: single-turn
- **Parser**: <e.g., custom
- **Rubric overview**: a reward of 1.0 is awarded if the model chooses the corret option, else it is 0

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval my-new-env
```

Configure model and sampling:

```bash
uv run vf-eval my-new-env   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

