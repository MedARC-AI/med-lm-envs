# MTSamples Replicate Benchmark

## Overview
- **Environment ID**: `mtsamples_replicate`
- **Short description**: Given patient notes with the PLAN section removed, generate a reasonable treatment plan. Evaluation adapted from HELM's MTSamples Replicate scenario.
- **Tags**: medical, clinical, single-turn, summarization, llm-judge, eval

## Datasets
- **Primary dataset**: MTSamples medical transcription (processed)
- **Source links**: [GitHub](https://github.com/raulista1997/benchmarkdata/tree/main/mtsamples_processed), [HELM scenario](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/mtsamples_replicate_scenario.py)
- **Split sizes**: Evaluation only (no predefined train/test splits)

## Task
- **Type**: Single-Turn
- **Rubric overview**: MultiJudgeRubric (LLM-as-a-Judge evaluation adapted from HELM's MTSamples Replicate annotator)
- **Task description**: Given patient notes with the **PLAN section removed** (while preserving SUMMARY and FINDINGS), generate a reasonable treatment plan.
- **Prompt**: "Here are information about a patient, return a reasonable treatment plan for the patient."
- **Evaluation dimensions**:
  - **Accuracy** (1–5): Does the response provide clinically appropriate and correct treatment guidance?
  - **Completeness** (1–5): Does the response cover the key aspects of care implied by the note?
  - **Clarity** (1–5): Is the response clearly written and well structured for clinical readability?

## Quickstart

Run an evaluation with default settings:

```bash
prime eval run mtsamples_replicate -m "openai/gpt-5-mini" -n 5 -s
```

Run a single-judge evaluation:

```bash
medarc-eval mtsamples_replicate -m "openai/gpt-5-mini" -n 5 --judge-model "openai/gpt-5-mini"
```

This environment supports multi-judge via the `medarc-verifiers` `MultiJudge` rubric. When multiple judge models are specified, each scores independently and the final reward is their mean. `judge_model`, `judge_base_url`, and `judge_api_key` are all list-valued and correspond positionally; if only one `judge_base_url` or `judge_api_key` is provided, it is used for all judge models.

Use configured default judges:

```bash
medarc-eval mtsamples_replicate -m "openai/gpt-5-mini" --judge-model "openai/gpt-5-mini" --judge-model "x-ai/grok-4.1-fast"
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `cache_dir` | str \| Path \| None | `~/.cache/medarc/mtsamples_replicate` | Local directory to cache downloaded datasets. |
| `use_think` | bool | `False` | Whether to use chain-of-thought prompting with `<think>...</think>` |
| `judge_model` | str \| list[str] | `"gpt-4o-mini"` | Model(s) used by the LLM judge |
| `judge_base_url` | str \| list[str] \| None | `None` | Custom API base URL(s) for judge model |
| `judge_api_key` | str \| list[str] \| None | `None` | API key(s) for judge model |

## Data Processing (HELM-aligned)

Following **HELM's MTSamples Replicate approach**:

1. **Section Extraction**: Extracts the first line following `PLAN:`, `SUMMARY:`, or `FINDINGS:` (priority order: `PLAN > SUMMARY > FINDINGS`)
2. **Input Cleaning**: Removes **only the `PLAN:` section** from the input text; all other sections (e.g., SUMMARY, FINDINGS, IMPRESSION) are preserved as clinical context.
3. **Reference**: The extracted section content is used as the gold reference answer.

## Dataset Example

**1-year-old Exam – H&P**

**Input (PLAN removed):**
```
Medical Specialty: Pediatrics - Neonatal
Sample Name: 1-year-old Exam - H&P
Description: Health maintenance exam for 1-year-old female.
...
IMPRESSION:
Routine well child care. Acute conjunctivitis.
```

**Reference Answer (PLAN):**
```
Diagnostic & Lab Orders: Ordered blood lead.
```

## Notes

- The `question` field contains the full patient note with the **PLAN section removed**
- The `answer` field contains the **first line** after the selected section header
- SUMMARY and FINDINGS may remain in the input, consistent with HELM's Replicate benchmark
- Scores are normalized to `[0, 1]` by averaging normalized dimension scores

## References

```bibtex
@misc{helm2023,
  title={Holistic Evaluation of Language Models},
  author={Liang, Percy and Bommasani, Rishi and Lee, Tony and others},
  year={2023},
  url={https://github.com/stanford-crfm/helm}
}
```
