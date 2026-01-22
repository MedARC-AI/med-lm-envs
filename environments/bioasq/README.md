# BioASQ

## Overview

- **Environment ID:** `bioasq`
- **Short description:**  
  BioASQ is a benchmark for large-scale biomedical semantic question answering that evaluates a model’s ability to generate comprehensive, expert-level answers grounded in scientific literature.
- **Task Type:**  
  Single-turn biomedical question answering and multi-document summarization.

---

## Dataset

BioASQ-QA reflects real-world information needs of biomedical experts. The benchmark is challenging because systems must reason over both structured and unstructured biomedical evidence to generate paragraph-sized *ideal answers*.

- **Split sizes:**
  - **Training:** 5,389 biomedical questions (compiled from previous BioASQ editions)
  - **Test:** 340 new questions across 4 batches (85 questions per batch)
  - **Total:** 5,729 questions with ideal answers and supporting evidence
- **Source:** https://huggingface.co/datasets/kroshan/BioASQ
- **Official Website:** http://bioasq.org/participate
- **Implementation based on:** BioASQ 2025 Overview Paper (Task 13b)

---

## Task

- **Type:** Single-turn
- **Rubric:** `JudgeRubric` (LLM-as-a-Judge evaluation adapted from official BioASQ manual assessment criteria)
- **Task description:**  
  Given a biomedical question and a set of relevant evidence snippets, generate a comprehensive *ideal answer*—a paragraph-sized summary intended for biomedical professionals.
- **Prompt template:**

  > *Here is a biomedical question and several relevant snippets. Provide a comprehensive answer strictly based on these snippets.*

---

## Evaluation Dimensions

Each response is scored on a **1–5 scale** (5 is best) following official BioASQ manual evaluation criteria:

- **Precision (1–5):**  
  Accuracy and relevance of biomedical facts. Unsupported information is penalized.
- **Recall (1–5):**  
  Coverage of all important biomedical concepts present in the gold answer.
- **Repetition (1–5):**  
  Redundancy in the response (1 = excessive repetition, 5 = none).
- **Readability (1–5):**  
  Clarity, organization, and professional readability.

---

## Quickstart

Run evaluation with default settings (dataset downloads automatically):

```bash
uv run vf-eval bioasq

### Use a custom judge model

```bash
uv run vf-eval bioasq -m gpt-4o --env-args '{"judge_model": "gpt-4o"}'
```

### Enable chain-of-thought (CoT) prompting

```bash
uv run vf-eval bioasq --env-args '{"use_think": true}'
```

### Notes

* Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
* By default, the environment uses **`gpt-4o-mini`** as the judge model.

---

## Environment Arguments

| Argument         | Type | Default         | Description                                                  |
| ---------------- | ---- | --------------- | ------------------------------------------------------------ |
| `cache_dir`      | str  | `None`          | Path to cache directory                                      |
| `use_think`      | bool | `False`         | Enable chain-of-thought prompting using `<think>...</think>` |
| `judge_model`    | str  | `"gpt-4o-mini"` | LLM used to judge ideal answers                              |
| `judge_base_url` | str  | `None`          | Custom base URL for judge API                                |
| `judge_api_key`  | str  | `None`          | API key for judge model                                      |

---

## Results Dataset Structure

### Core Evaluation Fields

* **`prompt`**
  Biomedical question and associated context snippets presented to the model.
  Represented as a list of message objects: `(role, content)`.

* **`completion`**
  Model-generated paragraph-sized ideal answer.

* **`reward`**
  Scalar score in the range **0.0–1.0**, computed as:

```text
(precision/5 + recall/5 + repetition/5 + readability/5) / 4
```

---

## Example Metadata (`info`)

The `info` field contains BioASQ-specific metadata for each example:

* **`filename`** – Unique question identifier
* **`question_type`** – One of: `yesno`, `factoid`, `list`, `summary`
* **`ideal_answer`** – Gold standard paragraph-sized reference answer
* **`exact_answer`** – Expected exact answer (format depends on question type)
* **`documents`** – List of relevant PubMed document URLs
* **`context`** – Concatenated text snippets used as supporting evidence
* **`judge_feedback`** – Judge explanations and per-dimension scores

### Notes

* The `question` field contains the biomedical question text.
* The `answer` field contains the gold ideal answer.
* Scores are normalized to 0–1 before averaging.
* If judge parsing fails, dimension scores default to `None` and are excluded.

---

## Data Processing

The dataset follows specifications from the **BioASQ 2025 paper**:

* **Question Types:** Yes/No, Factoid, List, Summary (Section 2.1)
* **Ideal Answers:** Paragraph-sized summaries for all question types
* **Exact Answers:**

  * Yes/No for yes-no questions
  * Entity name(s) for factoid and list questions
* **Supporting Evidence:** PubMed documents and curated text snippets

---

## Dataset Examples

### Example 1: Yes/No Question

**Question:**
Is the protein Papilin secreted?

**Question Type:** `yesno`

**Ideal Answer (Reference):**
Yes, Papilin is a secreted protein. It is an extracellular matrix glycoprotein that contains proteoglycan-like domains and is secreted into the basement membrane.

---

### Example 2: Factoid Question

**Question:**
Which gene is mutated in Huntington's disease?

**Question Type:** `factoid`

**Ideal Answer (Reference):**
Huntington's disease is caused by mutations in the HTT gene (also known as the Huntingtin gene). The mutation involves an expansion of CAG trinucleotide repeats in the HTT gene.

---

### Example 3: List Question

**Question:**
Which are the different isoforms of the mammalian Notch receptor?

**Question Type:** `list`

**Ideal Answer (Reference):**
In mammals, there are four Notch receptor isoforms: Notch1, Notch2, Notch3, and Notch4.

---

### Example 4: Summary Question

**Question:**
What is the mechanism of action of pembrolizumab?

**Question Type:** `summary`

**Ideal Answer (Reference):**
Pembrolizumab is a humanized monoclonal antibody that targets the programmed cell death protein 1 (PD-1) receptor. It blocks the interaction between PD-1 and its ligands PD-L1 and PD-L2, preventing pathway-mediated immune inhibition and enhancing anti-tumor immunity.

---

## Dataset Statistics

Breakdown based on the **BioASQ 2025 paper (Table 1)**:

| Dataset      | Questions | Yes/No    | List      | Factoid   | Summary   | Avg Docs | Avg Snippets |
| ------------ | --------- | --------- | --------- | --------- | --------- | -------- | ------------ |
| Training     | 5,389     | 1,459     | 1,047     | 1,600     | 1,283     | 9.74     | 12.78        |
| Test Batch 1 | 85        | 17        | 23        | 26        | 19        | 2.68     | 3.74         |
| Test Batch 2 | 85        | 17        | 19        | 27        | 22        | 2.71     | 3.06         |
| Test Batch 3 | 85        | 22        | 22        | 20        | 21        | 3.00     | 3.66         |
| Test Batch 4 | 85        | 26        | 19        | 22        | 18        | 3.15     | 3.92         |
| **Total**    | **5,729** | **1,541** | **1,130** | **1,695** | **1,363** | **9.33** | **12.23**    |

---

## References

```bibtex
@article{nentidis2025bioasq,
  title={Overview of BioASQ 2025: The thirteenth BioASQ challenge on large-scale biomedical semantic indexing and question answering},
  author={Nentidis, Anastasios and Katsimpras, Georgios and Krithara, Anastasia and others},
  journal={arXiv preprint arXiv:2508.20554},
  year={2025}
}

@article{tsatsaronis2015bioasq,
  title={An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition},
  author={Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos and others},
  journal={BMC Bioinformatics},
  volume={16},
  pages={138},
  year={2015}
}

@article{krithara2023bioasq,
  title={BioASQ-QA: A manually curated corpus for Biomedical Question Answering},
  author={Krithara, Anastasia and Nentidis, Anastasios and Bougiatiotis, Konstantinos and Paliouras, Georgios},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={170},
  year={2023}
}
```

