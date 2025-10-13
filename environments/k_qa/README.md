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
- **Parser**: `ClaimExtractorParser`.Uses `medarc_verifiers.JSONParser` to parse a `{"claims": [...]}` JSON from the extractor LLM output.

- **Rubric overview**:
  - `comprehensiveness`: fraction of must-have claims entailed by the model’s predicted claims.
  - `hallucination_rate`: fraction of predicted claims that contradict any gold claim (must-have or nice-to-have).
  - Both metrics rely on an NLI-style judge LLM.

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

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### How it works 
1. The llm model produces an answer to each `Question`.
2. Convert answers by the LLM into atomic statements 
3. `ClaimExtractorParser` prompts the `extractor_model` to extract atomic `claims` from the answer.
4. The `judge_model` evaluates entailment/contradiction between predicted claims and gold claims to compute:
   - must-have claims  by (`comprehensiveness`).
   - Contradictions against gold claims by (`hallucination_rate`).


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

