# ACI-Bench

## Overview
- **Environment ID**: `aci-bench`
- **Short description**: Convert doctor-patient dialogue into structured clinical notes.
- **Tags**: medical, clinical, dialogue, summarization,llm-judge, single-turn, train, eval, test

## Datasets
- **Primary dataset**: `ACI-Bench`
- **Source links**: [Paper](https://www.nature.com/articles/s41597-023-02487-3), [Github](https://github.com/wyim/aci-bench), [HF Dataset](https://huggingface.co/datasets/mkieffer/ACI-Bench-MedARC)
- **Split sizes**: 

| subset     | transcript_version | train | valid | test1 | test2 | test3 | Total |
| ---------- | ------------------ | ----- | ----- | ----- | ----- | ----- | ----- |
| aci        | asr                | 35    | 11    | 22    | 22    | 22    | 112   |
| aci        | asrcorr            | 35    | 11    | 22    | 22    | 22    | 112   |
| aci        | humantrans         | 0     | 0     | 0     | 0     | 0     | 0     |
| virtassist | asr                | 0     | 0     | 0     | 0     | 0     | 0     |
| virtassist | asrcorr            | 0     | 0     | 0     | 0     | 0     | 0     |
| virtassist | humantrans         | 20    | 5     | 10    | 10    | 10    | 55    |
| virtscribe | asr                | 12    | 4     | 8     | 8     | 8     | 40    |
| virtscribe | asrcorr            | 0     | 0     | 0     | 0     | 0     | 0     |
| virtscribe | humantrans         | 12    | 4     | 8     | 8     | 8     | 40    |
| ALL        | ALL                | 114   | 35    | 70    | 70    | 70    | 359   |


The dataset consists of different subsets capturing different clinical workflows:
1) ambient clinical intelligence (`aci`): doctor-patient dialogue
2) virtual assistant (`virtassist`): doctor-patient dialogue with queues to trigger Dragon Copilot, e.g., "hey, dragon. show me the chest x-ray"
3) virtual scribe (`virtscribe`): doctor-patient dialogue with a short dictation from the doctor about the patient at the very beginning

There are three different transcription versions:
1) `asr`: machine-transcribed
2) `asrcorr`: human corrections to `asr`, for example: "nonsmile" in D2N081 --> "non-small" in ACI006
3) `humantrans`: transcribed by a human

The subsets have the following transcription versions:
1) `aci`: `asr` and `asrcorr`
2) `virtassist`: `humantrans` only
3) `virtscribe`: `asr` and `humantrans`


## Task
- **Type**: single-turn
- **Rubric overview**: LLM-as-a-judge evaluation using prompts adapted from MedHELM (single or multi-judge)
- **Evaluation dimensions**:
  - **Accuracy** (1-5): Does the clinical note correctly capture the main medical issue and clinical details?
  - **Completeness** (1-5): Does the clinical note include all important medical information?
  - **Clarity** (1-5): Is the clinical note easy to understand for clinical use?

## Quickstart

Run a quick evaluation with `prime eval`:

```bash
prime eval run aci-bench -m "openai/gpt-5-mini" -n 5 -s
```

To pass environment-specific options, use `--env-args` (JSON).

```bash
prime eval run aci-bench -m "openai/gpt-5-mini" -n 5 --env-args '{"subset": "aci", "judge_model": "openai/gpt-5-mini"}'
```

Or use `medarc-eval` for named flags:

```bash
medarc-eval aci-bench -m "openai/gpt-5-mini" -n 5 --subset aci --judge-model "openai/gpt-5-mini"
```

This environment supports multi-judge via the `medarc-verifiers` `MultiJudge` rubric. When multiple judge models are specified, each scores independently and the final reward is their mean. `judge_model`, `judge_base_url`, and `judge_api_key` are all list-valued and correspond positionally; if only one `judge_base_url` or `judge_api_key` is provided, it is used for all judge models.

```bash
medarc-eval aci-bench -m "openai/gpt-5-mini" --judge-model "openai/gpt-5-mini" --judge-model "google/gemini-3-flash"
```

## Environment Arguments

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `subset`             | str  | `all`| The subset of the dataset to use (`all`, `aci`, `virtassist`, `virtscribe`)|
| `transcript_version` | str  | `all`| The transcript version to use (`all`, `asr`, `asrcorr`, `humantrans`)|
| `answer_format`      | str  | `xml` | The format of the answer (`xml`, `boxed`)|
| `system_prompt`      | str \| None  | `None` | Optional system prompt override |
| `judge_model`        | str \| list[str]  | `openai/gpt-5-mini` | Model identifier(s) for the LLM judge |
| `judge_base_url`     | str \| list[str]  | `None` | Custom API base URL(s) for judge model (defaults to OpenAI API) |
| `judge_api_key`      | str \| list[str]  | `None` | API key(s) for judge model. Falls back to `JUDGE_API_KEY` environment variable if not provided |


### Notes

- The `question` field in the dataset maps to the full conversation text
- The `answer` field contains the gold standard summary (also available as `reference_response` in `info`)
- Scores are normalized to 0-1 by dividing each dimension score (1-5) by 5 and averaging across dimensions
- If judge response parsing fails, dimension scores default to `None` and do not contribute to the final reward

## Dataset Examples

```
Dialogue:
[doctor] good morning julie how are you doing this morning
[patient] i've been better my primary care doctor wanted me to see you because of this this knee pain that i've been having for about six months now
...

Note:
CHIEF COMPLAINT
Bilateral knee pain.

SOCIAL HISTORY
The patient is an avid runner. She also works from home.
...
```
## References

```bibtex
@article{aci-bench,
  author = {Wen{-}wai Yim and
                Yujuan Fu and
                Asma {Ben Abacha} and
                Neal Snider and Thomas Lin and Meliha Yetisgen},
  title = {ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
  journal = {Nature Scientific Data},
  year = {2023}
}
```
