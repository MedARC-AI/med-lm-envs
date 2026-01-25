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
  - `+0.01` for abstaining with `2` (unsure) (configurable via `unsure_reward`)
  - `0.0` for incorrect classification or malformed answer

The model is presented with a medical question and an answer, then must judge:
- `0` = Answer is factual
- `1` = Answer is hallucinated
- `2` = Unsure (partial credit)

### Differences verses MedHallu paper

This environment intentionally differs from the MedHallu paper’s evaluation protocol:

- **We evaluate both options per item**: for each dataset row, we create two evaluation examples — one pairing the question with the **Ground Truth** answer (label `0`) and one pairing it with the **Hallucinated Answer** (label `1`). The paper’s implementation samples one of the two.
- **F1 is computed via postprocessing**: the paper reports **F1** (treating hallucination as the positive class). In this repo, you should compute F1 by postprocessing the `results.jsonl` output and dropping `\boxed{2}` (unsure) predictions.

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
| `difficulty` | str | `"all"` | Filter by difficulty: `"easy"`, `"medium"`, `"hard"`, or `"all"` |
| `use_knowledge` | bool | `False` | If `True`, includes the "Knowledge" field in the prompt as additional context |
| `unsure_reward` | float | `0.01` | Reward assigned when the model outputs `\boxed{2}` |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Scalar reward used by Verifiers (see rubric overview above) |
| `accuracy` | Exact match on the target label (`0` or `1`) |
| `precision/recall/f1` | Not produced by the environment directly; compute via postprocessing (below) |

### Postprocessing (F1)

After you run an eval, compute paper-style F1 (positive label `1`) and update the run’s `metadata.json`:

```bash
uv run python environments/medhallu/postprocess.py /path/to/results.jsonl
```

This script:
- extracts `\boxed{0|1|2}` from completions
- drops missing/malformed answers
- drops `\boxed{2}` (unsure)
- computes `accuracy`, `precision`, `recall`, `f1` (with `1` as the positive class)

### Hallucination Types
The model is trained to detect these hallucination categories:
- **Misinterpretation of Question**: Off-topic or irrelevant responses due to misunderstanding
- **Incomplete Information**: Pointing out what's false without providing correct information
- **Mechanism and Pathway Misattribution**: False attribution of biological mechanisms or disease processes
- **Methodological and Evidence Fabrication**: Invented research methods, statistics, or clinical outcomes
