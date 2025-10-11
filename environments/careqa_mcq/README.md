# careqa

Evaluation environment for the [HPAI-BSC/CareQA](https://huggingface.co/datasets/HPAI-BSC/CareQA) multiple-choice dataset.

### Overview
- **Environment ID**: `careqa_mcq`  
- **Short description**: CareQA is a healthcare QA dataset with **multiple-choice** and **open-ended clinical reasoning questions**. This environment is for the MCQs only.  
- **Tags**: healthcare, medical QA, clinical reasoning, MCQ, single-turn

### Datasets
- **Primary dataset(s)**:  
  - `CareQA_en` – multiple-choice clinical questions with 4 options and correct answer labels.  
- **Source links**:  
  - [Hugging Face CareQA dataset](https://huggingface.co/datasets/HPAI-BSC/CareQA)

### Task
- **Type**: single-turn  
- **Parser**: custom prompt mapping (no structured markup)  
- **Rubric overview**:  
**MCQ (`closed_mcq`)**: `vf.Rubric()` measuring **accuracy** (letter match).  

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval careqa
```

Configure model and sampling:

```bash
uv run vf-eval careqa_mcq --model gpt-4.1-mini --num-examples 3 -s
``` 


### Environment Arguments

| Arg            | Type | Default | Description |
|----------------|------|---------|-------------|
| `max_examples` | int  | `-1`    | Maximum number of examples to evaluate; use `-1` for full dataset |
| `split`        | str  | `"test"` | Dataset split to use: `train`, `validation`, or `test` |
| `verbose`      | bool | `False` | Print prompt/answer samples during evaluation |

---

### Metrics

| Metric        | Meaning |
|---------------|---------|
| `reward`      | Main scalar reward (weighted sum of rubric criteria) |
| `accuracy`    | Exact match on target MCQ answer (letter A–D) |


