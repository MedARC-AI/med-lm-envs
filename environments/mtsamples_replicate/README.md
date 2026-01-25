# MTSamples Replicate Benchmark

## Dataset

- **Source**:
  - [MTSamples medical transcription repository (processed)](https://github.com/raulista1997/benchmarkdata/tree/main/mtsamples_processed)
  - Implementation based on **HELM’s MTSamples Replicate scenario**  
    https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/mtsamples_replicate_scenario.py

---

## Task

- **Type**: Single-Turn
- **Rubric overview**: JudgeRubric (LLM-as-a-Judge evaluation adapted from HELM’s MTSamples Replicate annotator)
- **Task description**:  
  Given patient notes with the **PLAN section removed** (while preserving SUMMARY and FINDINGS),
  generate a reasonable treatment plan.
- **Prompt**:  
  > "Here are information about a patient, return a reasonable treatment plan for the patient."
- **Evaluation dimensions**:
  - **Accuracy** (1–5): Does the response provide clinically appropriate and correct treatment guidance?
  - **Completeness** (1–5): Does the response cover the key aspects of care implied by the note?
  - **Clarity** (1–5): Is the response clearly written and well structured for clinical readability?

---

## Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval mtsamples_replicate
```

Use a custom judge model:

```bash
uv run vf-eval mtsamples_replicate -m gpt-4.1-mini --env-args '{"judge_model": "gpt-5-mini"}'
```

Enable chain-of-thought prompting:

```bash
uv run vf-eval mtsamples_replicate --env-args '{"use_think": true}'
```

---

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `cache_dir` | str \| Path \| None | `~/.cache/mtsamples_procedures` | Local directory to cache downloaded datasets. |
| `use_think` | bool | `False` | Whether to use chain-of-thought prompting with `<think>...</think>` |
| `judge_model` | str | `"gpt-4o-mini"` | Model used by the LLM judge |
| `judge_base_url` | str \| None | `None` | Custom API base URL for judge model |
| `judge_api_key` | str \| None | `None` | API key for judge model |

---

## Data Processing (HELM-aligned)

Following **HELM’s MTSamples Replicate approach**:

1. **Section Extraction**  
   Extracts the first line following `PLAN:`, `SUMMARY:`, or `FINDINGS:`  
   (priority order: `PLAN > SUMMARY > FINDINGS`)
2. **Input Cleaning**  
   Removes **only the `PLAN:` section** from the input text; all other sections
   (e.g., SUMMARY, FINDINGS, IMPRESSION) are preserved as clinical context.
3. **Reference**  
   The extracted section content is used as the gold reference answer.

---

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

---

## Notes

- The `question` field contains the full patient note with the **PLAN section removed**
- The `answer` field contains the **first line** after the selected section header
- SUMMARY and FINDINGS may remain in the input, consistent with HELM’s Replicate benchmark
- Scores are normalized to `[0, 1]` by averaging normalized dimension scores

---

## References

```bibtex
@misc{helm2023,
  title={Holistic Evaluation of Language Models},
  author={Liang, Percy and Bommasani, Rishi and Lee, Tony and others},
  year={2023},