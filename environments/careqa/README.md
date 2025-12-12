# careqa

Evaluation environment for the [HPAI-BSC/CareQA](https://huggingface.co/datasets/HPAI-BSC/CareQA) dataset.

### Overview
- **Environment ID**: `careqa`  
- **Short description**: CareQA is a healthcare QA dataset with **multiple-choice** and **open-ended clinical reasoning questions**. This environment supports both modes through the `mode` parameter.  
- **Tags**: healthcare, medical QA, clinical reasoning, MCQ, single-turn

### Datasets
- **Primary dataset(s)**:
  - `CareQA_en` – multiple-choice clinical questions with 4 options and correct answer labels
  - `CareQA_en_open` – open-ended clinical questions with reference answers
- **Source links**:
  - [Hugging Face CareQA dataset](https://huggingface.co/datasets/HPAI-BSC/CareQA)

### Task
- **Type**: single-turn
- **Parser**:
  - MCQ mode: `vf.Parser()` or `vf.ThinkParser()` for extracting boxed answers
  - Open-ended mode: `XMLParser()` for judge responses
- **Rubric overview**:
  - **MCQ mode (`en`)**: `vf.Rubric()` measuring **accuracy** (letter match A–D)
  - **Open-ended mode (`open`)**: `vf.JudgeRubric()` using an LLM-as-judge to score free-text answers for correctness and clinical reasoning

### Quickstart

**Multiple-choice evaluation:**
```bash
medarc-eval careqa --mode en --model gpt-4.1-mini --num-examples 10 -s
```

**Open-ended evaluation:**
```bash
medarc-eval careqa --mode open --model gpt-4.1-mini --num-examples 10 -s
```

**With think-mode prompting (MCQ only):**
```bash
medarc-eval careqa --mode en --use-think --model gpt-4.1-mini --num-examples 10 -s
```

**With shuffled answer options (MCQ only):**
```bash
medarc-eval careqa --mode en --shuffle-answers --shuffle-seed 42 --model gpt-4.1-mini -n 10 -s
```

### Configuration Options

#### Common Parameters
- `--mode`: Select mode: `en` (multiple-choice) or `open` (open-ended). Default: `open`
- `--split`: Dataset split to use. Default: `test`
- `--system-prompt`: Custom system prompt (uses mode-appropriate default if not specified)

#### MCQ-Specific Parameters
- `--use-think`: Enable think-style prompting with boxed answers
- `--shuffle-answers`: Randomly shuffle answer options
- `--shuffle-seed`: Seed for answer shuffling (default: 1618)

#### Open-Ended-Specific Parameters
- `--judge-model`: Model for LLM-as-judge evaluation (default: `gpt-4o-mini`)
- `--judge-base-url`: Base URL for judge API
- `--judge-api-key`: API key for judge (falls back to `OPENAI_API_KEY` env var)

### Metrics

#### MCQ Mode
| Metric        | Meaning |
|---------------|---------|
| `reward`      | Main scalar reward (weighted sum of rubric criteria) |
| `accuracy`    | Exact match on target MCQ answer (letter A–D) |

#### Open-Ended Mode
| Metric        | Meaning |
|---------------|---------|
| `reward`      | Main scalar reward (weighted sum of rubric criteria) |
| `judge_score` | LLM-assigned score evaluating answer quality, correctness, and clinical reasoning |

### Example Usage

```python
import verifiers as vf

# Load MCQ environment
env_mcq = vf.load_environment("careqa", mode="en", shuffle_answers=True)

# Load open-ended environment
env_open = vf.load_environment("careqa", mode="open", judge_model="gpt-4o-mini")
```
