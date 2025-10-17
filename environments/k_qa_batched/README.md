# k_qa_batched (same env as k_qa but it evaluates both the comprehensiveness and hallucination in ONE API call )

> Replace the placeholders below, then remove this callout.

### Overview
- **Environment ID**: `k_qa_batched`
- **Short description**: An Evaluation for free form answers using concepts from FActScore ( break in to atomic statements) and calculating metrics for any LLM using comprehensiveness and hallucination in ONE API Call
- **Tags**: medllm, medical, FActScore

### Datasets
- **Primary dataset(s)**: Dataset used to evaluate is from K-QA paper with the dataset name "question_w_answers.jsonl"
- **Source links**: 
 - Paper: https://arxiv.org/abs/2401.14493
 - Dataset / Project: https://github.com/Itaymanes/K-QA
- **Split sizes**: 201 counts

### Task
- **Type**: single-turn
- **Parser**: Pass-through (`vf.Parser` with `extract_fn=lambda x: x`) — the environment reads model messages directly.
- **Rubric overview**:
  - A `RubricGroup` with two rubrics sharing state:
    1) **Extraction**: A `JudgeRubric` prompts the extractor LLM to decompose the model’s free-form answer into a JSON list of `claims`, stored in `state["kqa"]["claims"]`.
    2) **Scoring (single judge call)**: A second `JudgeRubric` evaluates:
       - **comprehensiveness**: fraction of must-have claims that are entailed by any generated claim.
       - **hallucination_count**: number of generated claims that contradict any gold claim (must-have or nice-to-have).
  - The scoring step is performed by a single prompt (`batch_eval_prompt`) that returns both signals in one response.


### How it works
1. The model answers a `Question` using a generation prompt.
2. The extractor rubric asks an LLM to output `{"claims": [...]}` that capture atomic facts from the model’s answer.
3. The scoring rubric runs one batched NLI-style evaluation to produce:
   - `comprehensiveness.entailed_must_have_claims`
   - `hallucination.contradictory_generated_claims`
4. Metrics are derived from these lists and recorded for the evaluation.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval k_qa_batched
```

Configure model and sampling:

```bash
uv uv run vf-eval k_qa_batched   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `extractor_model` | str | `"gpt-4.1-mini"` | LLM used to extract `claims` from the model’s free-form answer. |
| `judge_model` | str | `"gpt-4.1-mini"` | LLM used for batched NLI scoring (comprehensiveness + hallucination). |
| `extractor_base_url` | str or null | `None` | Optional custom base URL for the extractor client. |
| `extractor_api_key` | str or null | `None` | API key for the extractor client (falls back to env var, see Auth). |
| `judge_base_url` | str or null | `None` | Optional custom base URL for the judge client. |
| `judge_api_key` | str or null | `None` | API key for the judge client (falls back to env var, see Auth). |



### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted sum of criteria (by default: `comprehensiveness` + `hallucination_count`). |
| `comprehensiveness` | Fraction of must-have claims that are entailed by any generated claim (0.0–1.0; higher is better). |
| `hallucination_count` | Number of generated claims that contradict any gold claim (0, 1, 2, ...; lower is better). |

