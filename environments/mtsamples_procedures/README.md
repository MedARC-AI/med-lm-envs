# MTSamples Procedures

### Overview
- **Environment ID**: `mtsamples_procedures`
- **Short description**: MTSamples Procedures is a benchmark of medical transcription samples that tests a model's ability to generate reasonable treatment plans from patient procedure notes.

### Dataset
- **Split sizes**:
  - Evaluation: ~90 examples (all used for evaluation)
  - Note: This is an evaluation-only benchmark with no predefined train/test splits
- **Source**:
  - [MTSamples medical transcription repository](https://github.com/raulista1997/benchmarkdata/tree/main/mtsample_procedure)
  - Implementation based on [HELM's MTSamples Procedures scenario](https://github.com/stanford-crfm/helm/blob/51c3389f3820b940cca2fcb759dfe8f0b0160f46/src/helm/benchmark/scenarios/mtsamples_procedures_scenario.py)

### Task
- **Type**: Single-Turn
- **Rubric overview**: JudgeRubric (LLM-as-a-Judge evaluation adapted from HELM's MTSamples Procedures Annotator)
- **Task description**: Given patient notes (procedure note with PLAN/SUMMARY/FINDINGS sections removed), generate a reasonable treatment plan
- **Prompt**: "Here are information about a patient, return a reasonable treatment plan for the patient."
- **Evaluation dimensions**:
  - **Accuracy** (1-5): Does the response provide correct clinical advice that follows established clinical guidelines?
  - **Completeness** (1-5): Does the response include all important aspects of patient care mentioned in the reference?
  - **Clarity** (1-5): Is the response written clearly and organized in a way that is easy to read for clinicians?

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval mtsamples_procedures
```

Use a custom judge model:

```bash
uv run vf-eval mtsamples_procedures -m gpt-4.1-mini --env-args '{"judge_model": "gpt-5-mini"}'
```

Enable chain-of-thought prompting:

```bash
uv run vf-eval mtsamples_procedures --env-args '{"use_think": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The environment defaults to using `gpt-4o-mini` as the judge model.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `cache_dir` | str \| Path \| None | `~/.cache/mtsamples_procedures` | Local directory to cache downloaded datasets. Can also be set via `MTSAMPLES_PROCEDURES_CACHE_DIR` environment variable. |
| `use_think` | bool | `False` | Whether to use chain-of-thought prompting with `<think>...</think>` tags |
| `judge_model` | str | `"gpt-4o-mini"` | Model identifier for the LLM judge evaluating procedural plans |
| `judge_base_url` | str \| None | `None` | Custom API base URL for judge model (defaults to OpenAI API) |
| `judge_api_key` | str \| None | `None` | API key for judge model. Falls back to `JUDGE_API_KEY` environment variable if not provided |

### Results Dataset Structure
#### Core Evaluation Fields

- **`prompt`** - The patient notes presented to the model (list of message objects with `role` and `content`)
- **`completion`** - The model's generated treatment plan (list of message objects)
- **`reward`** - Overall score from 0.0 to 1.0, calculated as the average of normalized dimension scores: `(accuracy/5 + completeness/5 + clarity/5) / 3`

#### Example Metadata (`info`)
Contains all the MTSamples-specific information about each procedure:

- **`filename`** - Original filename from the GitHub repository
- **`extracted_section`** - Which section was used as reference ("PLAN", "SUMMARY", or "FINDINGS")
- **`procedure_note`** - The patient notes with sections removed (same as `question` field)
- **`reference_plan`** - Gold standard treatment plan/summary (same as `answer` field)
- **`judge_feedback`** - List of judge evaluations with scores and explanations for each dimension

#### Notes

- The `question` field contains everything BEFORE the first PLAN/SUMMARY/FINDINGS section (HELM's exact approach)
- The `answer` field contains the first line after the prioritized section header (PLAN > SUMMARY > FINDINGS)
- Scores are normalized to 0-1 by dividing each dimension score (1-5) by 5 and averaging across dimensions
- If judge response parsing fails, dimension scores default to `None` and do not contribute to the final reward

### Data Processing

Following HELM's exact approach:
1. **Section Extraction**: Extracts the first line after `PLAN:`, `SUMMARY:`, or `FINDINGS:` headers (priority order: PLAN > SUMMARY > FINDINGS)
2. **Input Cleaning**: Takes everything BEFORE the first section header found as the input
3. **Reference**: The extracted section content becomes the gold standard answer

### Dataset Examples

**Example: AC Separation Revision & Hardware Removal**
```
Patient Notes (input):
Medical Specialty:Orthopedic
Sample Name: AC Separation Revision & Hardware Removal
Description: Removal of the hardware and revision of right AC separation...
PREOPERATIVE DIAGNOSIS: Right AC separation.
POSTOPERATIVE DIAGNOSIS: Right AC separation.
PROCEDURES: Removal of the hardware and revision of right AC separation.
ANESTHESIA: General.
BLOOD LOSS: 100 cc.
COMPLICATIONS: None.

Reference Answer (extracted from SUMMARY section):
After informed consent was obtained and verified, the patient was brought to
the operating room and placed supine on the operating table. After uneventful
general anesthesia was obtained, he was positioned in the beach chair...

Generated Treatment Plan:
1. Postoperative Care: Monitor vital signs and surgical site for signs of infection...
2. Immobilization: Use of a sling or shoulder immobilizer for 2-4 weeks...
3. Physical Therapy: Begin passive range of motion exercises around 2-3 weeks post-op...
4. Follow-up: Schedule follow-up visits at 1-2 weeks post-op for wound check...
```

### References


**HELM MTSamples Implementation**
```bibtex
@misc{helm2023,
  title={Holistic Evaluation of Language Models},
  author={Liang, Percy and Bommasani, Rishi and Lee, Tony and others},
  year={2023},
  url={https://github.com/stanford-crfm/helm}
}
```
