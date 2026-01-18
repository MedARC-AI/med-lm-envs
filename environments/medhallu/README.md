# medhallu

### Overview
- **Environment ID**: `medhallu`
- **Short description**: Medical hallucination detection benchmark evaluating whether models can identify factual vs. hallucinated medical answers.
- **Tags**: hallucination-detection, medical, classification, single-turn

### Datasets Information

- **Paper:**: [MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models.](https://arxiv.org/abs/2502.14302)
- **Source links**: [UTAustin-AIHealth/MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)
- **Split sizes**: 
  - `pqa_labeled`: ~1k high-quality human-labeled examples
  - `pqa_artificial`: ~9k synthetically generated examples

### Task
- **Type**: single-turn
- **Parser**: Boxed answer parser (`extract_boxed_answer`) - expects answers in `\boxed{}`
- **Rubric overview**: 
  - `+1.0` for correct classification (matching target `0` or `1`)
  - `-0.01` for abstaining with `2` (unsure)
  - `-1.0` for incorrect classification

The model is presented with a medical question and an answer, then must judge:
- `0` = Answer is factual
- `1` = Answer is hallucinated
- `2` = Unsure (partial credit)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval medhallu
```

Configure model and sampling:

```bash
uv run vf-eval medhallu \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"subset": "pqa_labeled", "use_knowledge": false}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `subset` | str | `"pqa_labeled"` | Dataset subset: `"pqa_labeled"` (1k high-quality) or `"pqa_artificial"` (9k generated) |
| `use_knowledge` | bool | `False` | If `True`, includes the "Knowledge" field in the prompt as additional context |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward: +1.0 correct, -0.01 unsure, -1.0 incorrect |
| `accuracy` | Exact match on target answer (0 or 1) |

### Hallucination Types
The model is trained to detect these hallucination categories:
- **Misinterpretation of Question**: Off-topic or irrelevant responses due to misunderstanding
- **Incomplete Information**: Pointing out what's false without providing correct information
- **Mechanism and Pathway Misattribution**: False attribution of biological mechanisms or disease processes
- **Methodological and Evidence Fabrication**: Invented research methods, statistics, or clinical outcomes
