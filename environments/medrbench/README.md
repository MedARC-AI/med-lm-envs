# MedRBench

### Overview
- **Environment ID**: `medrbench`
- **Short description**: Medical reasoning benchmark for diagnosis and treatment planning on rare disease cases.
- **Tags**: medical, diagnosis, treatment, rare-disease, llm-judge, single-turn, multi-turn, eval

### Datasets
- **Primary dataset**: MedRBench
- **Source links**: [Paper](https://arxiv.org/abs/2402.09764), [GitHub](https://github.com/MAGIC-AI4Med/MedRBench)
- **Split sizes**:

| Split | Cases | Rare Disease Cases |
|-------|-------|-------------------|
| diagnosis | 957 | 491 |
| treatment | 496 | 165 |

### Task
- **Type**: diagnosis supports `oracle`, `1turn`, and `free_turn`; treatment is `oracle` only
- **System Prompt**: `"You are a professional doctor"` (matching original)
- **Parser**: Oracle/treatment use `### Answer:`; `1turn`/`free_turn` diagnosis use `### Conclusion:`
- **Rubric overview**: JudgeRubric (LLM-as-a-Judge evaluation using original MedRBench prompts)
- **Evaluation metric**: Binary accuracy (Correct/Wrong)

#### Diagnosis Split (`outcome_accuracy`)
The model is given a clinical case summary and must provide the final diagnosis. The LLM judge uses the original MedRBench `acc_diagnose.txt` prompt to evaluate if the predicted diagnosis matches the ground truth, accounting for:
- Disease aliases (e.g., "Heart disease" = "Cardiac disease")
- Language variations (e.g., "heart attack" = "myocardial infarction")
- Partial matches where additional complications are mentioned

#### Treatment Split (`treatment_final_accuracy`)
The model is given a clinical case summary and must provide a treatment recommendation. The LLM judge uses the original MedRBench `acc_treatment_plan.txt` prompt to evaluate if the predicted treatment is clinically appropriate, considering:
- Semantic equivalence between predicted and ground truth treatments
- Valid alternative treatment approaches
- Additional care measures that don't contradict the main treatment

### Quickstart
Run an evaluation with default settings (all splits combined):

```bash
uv run vf-eval medrbench
```

Configure model, split, and other options:

```bash
# All splits combined (default)
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"judge_model": "gpt-4o"}'

# Diagnosis split only
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"split": "diagnosis", "judge_model": "gpt-4o"}'

# Treatment split only
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"split": "treatment", "judge_model": "gpt-4o"}'

# Rare disease cases only
uv run vf-eval medrbench \
    -m o3-mini \
    -n -1 \
    -a '{"split": "diagnosis", "rare_disease_only": true, "judge_model": "gpt-4o"}'

# 1-turn diagnosis mode
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"split": "diagnosis", "task_mode": "1turn", "judge_model": "gpt-4o"}'

# Free-turn diagnosis mode (max 5 turns)
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"split": "diagnosis", "task_mode": "free_turn", "max_turns": 5, "judge_model": "gpt-4o"}'
```

### Environment Arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `split` | str | `all` | Dataset split: `diagnosis`, `treatment`, or `all` (default) |
| `rare_disease_only` | bool | `False` | If True, only include cases with rare diseases |
| `task_mode` | str | `oracle` | Diagnosis mode: `oracle`, `1turn`, or `free_turn` (diagnosis only) |
| `max_turns` | int | `5` | Max turns for `free_turn` diagnosis |
| `judge_model` | str | `gpt-5-mini` | Model identifier for the LLM judge (original uses `gpt-4o-2024-11-20`) |
| `judge_base_url` | str | `None` | Custom API base URL for judge model |
| `judge_api_key` | str | `None` | API key for judge model. Falls back to `JUDGE_API_KEY` or `OPENAI_API_KEY` environment variables |
| `patient_agent_model` | str | `gpt-4o` | Model identifier for the patient agent in multi-turn modes |
| `patient_agent_base_url` | str | `None` | Custom API base URL for the patient agent |
| `patient_agent_api_key` | str | `None` | API key for the patient agent (defaults to judge credentials) |
| `system_prompt` | str | `"You are a professional doctor"` | System prompt (matches original MedRBench) |

### Dataset Size Notes

This environment evaluates against the full dataset for the selected `split`. Use `-n` in `vf-eval` to subsample.

### Notes

- The `question` field contains the formatted clinical case with task instructions
- The `answer` field contains the ground truth diagnosis or treatment plan (also available as `reference_response` in `info`)
- Judge prompts are taken directly from MedRBench's original evaluation prompts (`acc_diagnose.txt` and `acc_treatment_plan.txt`)
- Treatment web-search evidence from the original implementation is not used; the judge prompt’s `[Additional Information]` section is left empty.
- Reward is binary: 1.0 for correct, 0.0 for incorrect (following original logic: `'correct' in evaluation_result.lower()`)
- Treatment remains oracle-only; `task_mode` applies to diagnosis split only
- Case metadata (body_category, disorder_category, checked_rare_disease) is available in `info` for analysis
- Data is loaded directly from the MedRBench GitHub repository
- Additional information web search removed from treatment judge prompt

### Dataset Examples

**Diagnosis Example:**
```
Case Summary:
- Patient Information: 13-year-old male
- Chief Complaint: Severe left eye pain
- History of Present Illness: Eyelid edema, erythema, localized warmth...
- Physical Examination: Febrile (39.5°C), signs of orbital inflammation
- Laboratory Findings: Elevated CRP, leukocytosis, neutrophilia
- Imaging: NCCT and NCMRI showing maxillary sinusitis and epidural empyema

Ground Truth Diagnosis:
Orbital cellulitis secondary to acute sinusitis with epidural empyema
```

**Treatment Example:**
```
Case Summary:
- Patient Information: 58-year-old man
- Chief Complaint: Cough worsening over 1 week
- History: Primary myelofibrosis with interstitial pneumonia
- Laboratory: Anemia (Hb 81.0 g/L), elevated LDH

Ground Truth Treatment:
JAK2 inhibitor therapy (ruxolitinib)
```

### References

```bibtex
@article{medrbench2024,
  title={MedRBench: A Medical Reasoning Benchmark for Large Language Models},
  author={MAGIC-AI4Med},
  journal={arXiv preprint arXiv:2402.09764},
  year={2024}
}
```

### Authors
This environment has been put together by:
- [Hunar Batra](https://github.com/hunarbatra)
- [Benjamin Warner](https://github.com/warner-benjamin)