# careqa_openended

Evaluation environment for the [HPAI-BSC/CareQA](https://huggingface.co/datasets/HPAI-BSC/CareQA) openended dataset.

### Overview
- **Environment ID**: `careqa_openended`  
- **Short description**: CareQA is a healthcare QA dataset with **multiple-choice** and **open-ended clinical reasoning questions**. This environment is for the open-ended questions only.  
- **Tags**: healthcare, medical QA, clinical reasoning, single-turn

### Datasets
- **Primary dataset(s)**:  
  - `CareQA_en_open` – open-ended clinical questions with reference answers.
- **Source links**:  
  - [Hugging Face CareQA dataset](https://huggingface.co/datasets/HPAI-BSC/CareQA)

### Task
- **Type**: single-turn  
- **Parser**: custom prompt mapping (no structured markup)  
- **Rubric overview**:  
**Open-ended (`open_clinical`)**: `vf.JudgeRubric()` using an LLM-as-judge to score free-text answers for correctness and clinical reasoning. 

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval careqa
```

Configure model and sampling:

```bash
uv run vf-eval careqa_openended --model gpt-4.1-mini --num-examples 3 -s
``` 

### Metrics

| Metric        | Meaning |
|---------------|---------|
| `reward`      | Main scalar reward (weighted sum of rubric criteria) |
|  `judge_score` | For open-ended questions, LLM-assigned score evaluating answer quality, correctness, and clinical reasoning |


