# k_qa

### Overview
- **Environment ID**: `k_qa`
- **Short description**: An Evaluation for free form answers using concepts from FActScore ( break in to atomic statements) and calculating metrics for any LLM using comprehensiveness and hallucination.

- **Tags**: medllm, medical, FActScore 

### Datasets
- **Primary dataset(s)**: Dataset used to evaluate is from K-QA paper with the dataset name "question_w_answers.jsonl"
- **Source links**: 
 - Paper: https://arxiv.org/abs/2401.14493
 - Dataset / Project: https://github.com/Itaymanes/K-QA
- **Split sizes**: 201 counts

### Task
- **Type**: single-turn
- **Parser**: Uses `medarc_verifiers.JSONParser` to parse a `{"claims": [...]}` JSON from the extractor LLM output.

- **Rubric overview**:
  - A `RubricGroup` with two phases that share state:
    1.  **Extraction**: A `JudgeRubric` extracts claims from the model's free-form answer and stores them in `state`.
    2.  **Scoring**: A second `JudgeRubric` reads the claims from `state` and computes:
        - `comprehensiveness`: fraction of must-have claims entailed by the model’s predicted claims.
        - `hallucination_rate`: fraction of predicted claims that contradict any gold claim (must-have or nice-to-have).
  - Both phases rely on an NLI-style judge LLM.

### Quickstart

Make sure you have an OpenAI API key available to the process:
```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Run an evaluation with defaults:
```bash
uv run vf-eval k_qa
```

Configure the generation model and sampling (example):
```bash
uv run vf-eval k_qa \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Override environment arguments (see table below):
```bash
uv run vf-eval k_qa \
  -a '{"extractor_model":"gpt-4-mini","judge_model":"gpt-4-mini"}'
```

To run a batched evaluation(default is set to false):
```bash
uv run vf-eval k_qa -a '{"batch": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### How it works 
1. The LLM agent produces a free-form answer to a `Question`.
2. An `extractor` rubric prompts the `extractor_model` to decompose the answer into atomic `claims`. These are stored internally.
3. A `scorer` rubric then uses the `judge_model` to evaluate entailment and contradiction between the model's claims and the gold claims to compute metrics.
   - `comprehensiveness` is calculated based on how many "must have" gold claims are entailed by the model's claims.
   - `hallucination_rate` is calculated based on how many of the model's claims contradict any of the gold claims.



### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `extractor_model` | str | `"gpt-4-mini"` | The model used to extract claims from the free form answer. |
| `judge_model` | str | `gpt-4-mini` | The model used for NLI-style scoring (entailment and contradiction) |
| `batch` | bool | `False` | Whether to run evaluation in a single batch call to the judge model. |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | The primary reward signal, which is equivalent to comprehensiveness |
| `comprehensiveness` | The fraction of "must-have" gold claims that are entailed by the claims made in the generated answer. A value of 1.0 means all essential information was covered |
| `hallucination_rate` | The count of claims in the generated answer that contradict any of the gold standard claims ("must-have" or "nice-to-have"). A lower value is better|

