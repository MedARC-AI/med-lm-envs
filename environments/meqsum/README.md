# MeQSum - Consumer Health Question Summarization

Evaluation environment for consumer health question summarization: condensing verbose patient questions into concise summaries.

### Overview
- **Environment ID**: `meqsum`
- **Short description**: Consumer health question summarization benchmark using the MeQSum dataset. This environment evaluates how well models can summarize verbose patient health queries into concise, focused questions.
- **Tags**: medical, nlp, summarization, single-turn, llm-judge, nlg-metrics
- **System Prompt**: "Summarize the patient health query into one question of 15 words or less."

---

### Dataset
- **Source**: [medarc/MeQSum-patient-consumer-health-questions](https://huggingface.co/datasets/medarc/MeQSum-patient-consumer-health-questions)
- **Based on**: MeQSum corpus from [Ben Abacha & Demner-Fushman, ACL 2019](https://aclanthology.org/P19-1215/) - "On the Summarization of Consumer Health Questions"
- **Split sizes**:
  - **Train:** 1,000 examples
  - **Validation:** 50 examples
  - **Test:** 150 examples
- **Task**: Given a verbose consumer health question, generate a concise summary (≤15 words).

---

### Task
- **Type:** Single-Turn Summarization
- **Input:** Consumer health question (verbose patient query)
- **Output:** Concise question summary (≤15 words)
- **Evaluation:** Dual evaluation approach following the Nature Medicine paper:
  1. **LLM-as-Judge**: Evaluates correctness, completeness, and conciseness (1-5 scale each)
  2. **Automatic Metrics**: BLEU, ROUGE (1/2/L/Lsum), BERTScore (precision/recall/F1)

The LLM-as-judge criteria are adapted from the **Nature Medicine reader study** (Methods section):
- **Correctness**: "Which summary includes less false information?" — evaluates precision (penalizes fabricated information)
- **Completeness**: "Which summary more completely captures important information?" — evaluates recall (important detail retained)
- **Conciseness**: "Which summary contains less non-important information?" — evaluates brevity (penalizes superfluous information)

The implementation follows the pattern established in `medicationqa`, using multi-dimensional scoring with a JSONParser.

---

### Quickstart

**Basic evaluation with default settings:**
```bash
python -m medarc_verifiers.cli.main meqsum -m gpt-4.1-mini -n 5 -r 1 --judge-model gpt-4.1-mini -s
```

**Run on validation split:**
```bash
python -m medarc_verifiers.cli.main meqsum --split validation -m gpt-4.1-mini -n 10 -r 1 --judge-model gpt-4.1-mini -s
```

**Fast evaluation (without automatic metrics):**
```bash
python -m medarc_verifiers.cli.main meqsum -m gpt-4.1-mini -n 10 -r 1 --judge-model gpt-4.1-mini --no-compute-auto-metrics -s
```

**Using a local model (e.g., Ollama):**
```bash
python -m medarc_verifiers.cli.main meqsum \
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
| `reward` | Normalized LLM-judge score (0-1), averaged across correctness, completeness, and conciseness |

#### LLM-Judge Dimensions (1-5 scale)

Criteria adapted from [Van Veen et al., Nature Medicine 2024](https://doi.org/10.1038/s41591-024-02855-5) (Methods - Reader study):

| Dimension | Description |
|-----------|-------------|
| `correctness` | Does the summary include false information? Evaluates precision—penalizes fabricated or incorrect information. |
| `completeness` | Does the summary completely capture important information? Evaluates recall—important detail retained. |
| `conciseness` | Does the summary contain non-important information? Evaluates brevity—penalizes superfluous information. Compares output length to reference. |

#### Automatic Metrics
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
- **`prompt`** – The consumer health question presented to the model.
- **`completion`** – The model-generated summary.
- **`reward`** – Normalized LLM-judge score in `[0, 1]`.

#### Example Metadata (`info`)
- **`idx`** – Original dataset index.
- **`original_question`** – The input consumer health question text.
- **`judge_feedback`** – Detailed LLM-judge evaluation with scores and reasoning.
- **`auto_metrics`** – Dictionary containing BLEU, ROUGE, and BERTScore values.
- **`length_metrics`** – Dictionary with model_length, reference_length, and length_ratio.

---

### References

**Dataset Source**
- Ben Abacha, A. & Demner-Fushman, D. "On the Summarization of Consumer Health Questions." *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 2228–2234, 2019. https://aclanthology.org/P19-1215/

**Evaluation Methodology**
- Van Veen, D., Van Uden, C., Blankemeier, L. et al. "Adapted large language models can outperform medical experts in clinical text summarization." *Nature Medicine* 30, 1134–1142 (2024). https://doi.org/10.1038/s41591-024-02855-5

**Evaluation Metrics**
- BLEU: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (ACL 2002)
- ROUGE: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (ACL 2004)
- BERTScore: Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (ICLR 2020)
