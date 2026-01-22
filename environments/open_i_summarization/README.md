# Open-I Summarization

Evaluation environment for radiology report summarization: generating impressions from findings.

### Overview
- **Environment ID**: `open_i_summarization`
- **Short description**: Radiology findings-to-impression summarization benchmark using the Open-I chest X-ray dataset. This environment evaluates how well models can distill radiology findings into concise, clinically accurate impressions.
- **Tags**: medical, radiology, summarization, single-turn, llm-judge, nlg-metrics
- **System Prompt**: "Summarize the radiology report findings into an impression with minimal text."

---

### Dataset
- **Source**: [medarc/open-i-summarization](https://huggingface.co/datasets/medarc/open-i-summarization)
- **Based on**: Indiana University Chest X-ray Collection (Open-I), as used in [Van Veen et al., Nature Medicine 2024](https://www.nature.com/articles/s41591-024-02855-5) - "Adapted large language models can outperform medical experts in clinical text summarization"
- **Split sizes**:
  - **Train:** 2,735 examples
  - **Validation:** 341 examples
  - **Test:** 343 examples
- **Task**: Given radiology findings, generate a concise clinical impression.

---

### Task
- **Type:** Single-Turn Summarization
- **Input:** Radiology findings (chest X-ray report text)
- **Output:** Clinical impression (summary)
- **Evaluation:** Dual evaluation approach following the Nature Medicine paper:
  1. **LLM-as-Judge**: Evaluates accuracy, completeness, and conciseness (1-5 scale each)
  2. **Automatic Metrics**: BLEU, ROUGE (1/2/L/Lsum), BERTScore (precision/recall/F1)

The LLM-as-judge implementation follows the pattern established in `medicationqa`, using multi-dimensional scoring with a JSONParser.

---

### Quickstart

**Basic evaluation with default settings:**
```bash
python -m medarc_verifiers.cli.main open_i_summarization -m gpt-4.1-mini -n 5 -r 1 --judge-model gpt-4.1-mini -s
```

**Example output:**
```
--- Evaluation ---
Environment: open_i_summarization
Model: gpt-4.1-mini
Provider: https://api.openai.com/v1/
Examples: 5
Rollouts per example: 1
--- Example ---
╭──────────────────────────────────── Step 0 ────────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓  │
│ ┃ Prompt                         ┃ Completion                      ┃ Reward ┃  │
│ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩  │
│ │ system: Summarize the          │ assistant: Normal heart size;   │   1.00 │  │
│ │ radiology report findings into │ no acute pulmonary findings;    │        │  │
│ │ an impression with minimal     │ mediastinal calcification and   │        │  │
│ │ text.                          │ dense right upper lung nodule   │        │  │
│ │                                │ consistent with prior           │        │  │
│ │ user: Heart size within normal │ granulomatous disease.          │        │  │
│ │ limits. No focal alveolar      │                                 │        │  │
│ │ consolidation, no definite     │                                 │        │  │
│ │ pleural effusion seen...       │                                 │        │  │
│ └────────────────────────────────┴─────────────────────────────────┴────────┘  │
╰────────────────────────────────────────────────────────────────────────────────╯
--- All ---
Rewards:
reward: avg - 1.000, std - 0.000
r1: [1.0, 1.0, 1.0, 1.0, 1.0]
```

**Run on validation split:**
```bash
python -m medarc_verifiers.cli.main open_i_summarization --split validation -m gpt-4.1-mini -n 10 -r 1 --judge-model gpt-4.1-mini -s
```

**Fast evaluation (without automatic metrics):**
```bash
python -m medarc_verifiers.cli.main open_i_summarization -m gpt-4.1-mini -n 10 -r 1 --judge-model gpt-4.1-mini --no-compute-auto-metrics -s
```

**Using a local model (e.g., Ollama):**
```bash
python -m medarc_verifiers.cli.main open_i_summarization \
  -m llama3 \
  --api-base-url http://localhost:11434/v1 \
  --env-args '{"judge_model":"llama3","judge_base_url":"http://localhost:11434/v1","judge_api_key":"ollama"}' \
  -n 5 -r 1 -s
```

---

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | `str` | `"test"` | Dataset split to use (`train`, `validation`, `test`). |
| `judge_model` | `str` | `"gpt-4o-mini"` | Model identifier for the LLM judge. |
| `judge_base_url` | `str \| None` | `None` | Custom API base URL (e.g., for Ollama or local models). |
| `judge_api_key` | `str \| None` | `None` | API key for the judge model (falls back to `OPENAI_API_KEY`). |
| `compute_auto_metrics` | `bool` | `True` | Whether to compute BLEU/ROUGE/BERTScore metrics. |
| `system_prompt` | `str \| None` | `None` | Custom system prompt (uses default if not provided). |

---

### Metrics

#### Primary Metric (Reward)
| Metric | Meaning |
|--------|---------|
| `reward` | Normalized LLM-judge score (0-1), averaged across accuracy, completeness, and conciseness |

#### LLM-Judge Dimensions (1-5 scale)
| Dimension | Description |
|-----------|-------------|
| `accuracy` | Does the impression correctly reflect the key findings without errors or hallucinations? |
| `completeness` | Does the impression capture all clinically significant findings? |
| `conciseness` | Is the impression appropriately brief while conveying essential information? |

#### Automatic Metrics (following Van Veen et al.)
| Metric | Description |
|--------|-------------|
| `bleu` | BLEU score (n-gram precision) |
| `rouge1` | ROUGE-1 (unigram overlap) |
| `rouge2` | ROUGE-2 (bigram overlap) |
| `rougeL` | ROUGE-L (longest common subsequence) |
| `rougeLsum` | ROUGE-Lsum (sentence-level LCS) |
| `bertscore_precision` | BERTScore precision |
| `bertscore_recall` | BERTScore recall |
| `bertscore_f1` | BERTScore F1 |

---

### Results Dataset Structure

#### Core Evaluation Fields
- **`prompt`** – The radiology findings presented to the model.
- **`completion`** – The model-generated impression.
- **`reward`** – Normalized LLM-judge score in `[0, 1]`.

#### Example Metadata (`info`)
- **`idx`** – Original dataset index.
- **`findings`** – The input radiology findings text.
- **`judge_feedback`** – Detailed LLM-judge evaluation with scores and reasoning.
- **`auto_metrics`** – Dictionary containing BLEU, ROUGE, and BERTScore values.

---

### Example Results

```
=== Example 1 ===
Model output: Normal heart size; no acute pulmonary findings; mediastinal 
              calcification and dense right upper lung nodule consistent 
              with prior granulomatous disease.
Reference: No acute cardiopulmonary findings
Reward: 1.0

LLM-as-Judge Scores:
  accuracy: 5/5
  completeness: 5/5
  conciseness: 5/5

Automatic Metrics:
  BLEU: 0.0000
  ROUGE-1: 0.2500
  ROUGE-L: 0.2500
  BERTScore F1: 0.8507
```

---

### References

**Dataset Source**
- Van Veen, D., Van Uden, C., Blankemeier, L. et al. "Adapted large language models can outperform medical experts in clinical text summarization." *Nature Medicine* 30, 1134–1142 (2024). https://doi.org/10.1038/s41591-024-02855-5

**Open-I Dataset**
```bibtex
@article{demner2016preparing,
  title={Preparing a collection of radiology examinations for distribution and retrieval},
  author={Demner-Fushman, Dina and Kohli, Marc D and Rosenman, Marc B and Shooshan, Sonya E and Rodriguez, Laritza and Antani, Sameer and Thoma, George R and McDonald, Clement J},
  journal={Journal of the American Medical Informatics Association},
  volume={23},
  number={2},
  pages={304--310},
  year={2016},
  publisher={Oxford University Press}
}
```

**Evaluation Metrics**
- BLEU: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (ACL 2002)
- ROUGE: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (ACL 2004)
- BERTScore: Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (ICLR 2020)
