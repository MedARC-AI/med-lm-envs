# LongHealth

## Overview
- **Environment ID**: `longhealth`
- **Short description**: Evaluates LLM ability to process and extract information from long clinical documents (5K-6.7K words per case)
- **Tags**: medical, clinical, single-turn, multiple-choice, long-context, eval


## Datasets
- **Paper**: https://arxiv.org/abs/2401.14490
- **Dataset / Project**: https://github.com/kbressem/LongHealth
- **Split sizes**: 400 questions total (20 patients × 20 questions each). No official train/val splits
- **Context length**: 5,090 to 6,754 words per patient case

## Benchmarks Tasks

### Task 1: Information Extraction
- Tests ability to extract correct information from long clinical documents
- Answer is ALWAYS present in provided documents
- 5-option MCQ (A/B/C/D/E)

### Task 2: Negation Detection & Hallucination Prevention
- Tests ability to identify when information is NOT available
- Creates pairs of examples:
  - **Negation**: Only distractor documents (should answer F: "Cannot be answered")
  - **Identification**: Answer docs + distractors (should answer correctly)
- 6-option MCQ (A/B/C/D/E/F)

### Task 3: Temporal Reasoning
- Embedded within Task 2 framework
- Questions focus on chronological ordering and temporal relationships
- Tests understanding of event sequences in medical timelines

## Quickstart
Run an evaluation with default settings (Task 1, first 10 examples):

```bash
prime eval run longhealth -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and task:

```bash
medarc-eval longhealth -m "openai/gpt-5-mini" -n 20 -s --task task2

medarc-eval longhealth -m "openai/gpt-5-mini" -n 10 -s --task all --doc-shuffle-seed 2718 --max-context-tokens 30000
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `task` | str | `"task1"` | Which task(s): `"task1"` (extraction), `"task2"` (negation), `"all"` (both) |
| `max_context_tokens` | int | `16000` | Maximum tokens for document context |
| `shuffle_docs` | bool | `True` | Shuffle document order to test positional bias |
| `doc_shuffle_seed` | int \| None | `-1` | Seed for document shuffling (`-1` for nondeterministic order each run) |
| `shuffle_answers` | bool | `False` | Shuffle answer options |
| `shuffle_seed` | int \| None | `1618` | Seed for answer shuffling |
| `max_examples` | int | `-1` | Limit number of examples (-1 for all) |

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Exact match accuracy (1.0 if correct letter, 0.0 otherwise) |
| `info.longhealth_task` | Which sub-task: `task1`, `task2_negation`, or `task2_identification` |
| `info.has_answer_docs` | Whether answer-containing documents were included |
| `info.num_docs` | Number of documents in the context |

## Example Usage

```python
import verifiers as vf

# Load Task 1
env = vf.load_environment("longhealth", task="task1")

# Load Task 2 with custom settings
env = vf.load_environment(
    "longhealth",
    task="task2",
    max_context_tokens=14000,
    shuffle_docs=True
)

# Run evaluation programmatically
from openai import AsyncOpenAI
client = AsyncOpenAI()
results = await env.evaluate(client, "gpt-4.1-mini", num_examples=10)
```

## Authors
This environment has been put together by:

Shamus Sim Zi Yang - ([@ss8319](https://github.com/ss8319))
