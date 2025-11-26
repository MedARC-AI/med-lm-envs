# MedicationQA

### Overview
- **Environment ID**: `medicationqa`
- **Short description**: MedicationQA (MedInfo 2019) is a benchmark of single-turn, consumer-style medication questions and expert answers. It evaluates how well models can provide safe, accurate, and complete responses to medication-related inquiries.

---

### Dataset
- **Source**: [Medication_QA_MedInfo2019](https://github.com/abachaa/Medication_QA_MedInfo2019)
- **Publication**: Abacha & Demner-Fushman, *MedInfo 2019* – “A Question-Answering Dataset for Medication Safety”
- **Split sizes**:
  - **Test:** 690  _(full dataset; no official train/validation partitions)_
- **Notes:**  
  The original dataset does not define official splits.  
  For consistency with other single-turn environments (e.g., `med_dialog`), this environment exposes the *entire* dataset as the `test` split.

---

### Task
- **Type:** Single-Turn QA  
- **Rubric:** LLM-as-a-Judge (adapted from MedHELM / MedDialog)  
- **Evaluation dimensions:**
  - **Accuracy (1–5)** – factual correctness of the answer  
  - **Completeness (1–5)** – inclusion of all relevant medication details  
  - **Clarity (1–5)** – readability and understandability for lay users  

---

### Quickstart

Run an evaluation with the default (OpenAI) judge:

```bash
uv run vf-eval medicationqa -m gpt-4o --num-examples 3 --save-results
```

Use a local Ollama model (e.g. `llama3`) for both answering and judging:

```bash
uv run vf-eval medicationqa \
  -m llama3 \
  --api-base-url http://localhost:11434/v1 \
  --env-args '{"judge_model":"llama3","judge_base_url":"http://localhost:11434/v1","judge_api_key":"ollama"}' \
  --num-examples 3 \
  --save-results
```

**Notes**
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.  
- The environment defaults to using `"gpt-4o-mini"` as the judge model.  

---

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `cache_dir` | `str \| Path \| None` | `~/.cache/medicationqa` | Local directory to cache the MedicationQA dataset. |
| `judge_model` | `str` | `"gpt-4o-mini"` | Model identifier for the LLM judge. |
| `judge_base_url` | `str \| None` | `None` | Custom API base URL (e.g. for Ollama or local models). |
| `judge_api_key` | `str \| None` | `None` | API key for the judge model (falls back to `JUDGE_API_KEY`). |

---

### Results Dataset Structure

#### Core Evaluation Fields
- **`prompt`** – The medication question presented to the model (either as plain text or a list of chat-style message objects with `role` and `content`).  
- **`completion`** – The model-generated answer, represented as a list of message objects (e.g., `[{"role": "assistant", "content": "..."}]`).  
- **`reward`** – Normalized score in `[0, 1]`, computed as the average of the three dimension scores: `(accuracy/5 + completeness/5 + clarity/5) / 3`.


#### Example Metadata (`info`)
- **`id`** – Unique identifier for each question (e.g. `medicationqa_42`).  
- **`question`** – Consumer-style medication question.  
- **`reference_answer`** – Gold reference answer from the MedInfo dataset.  
- **`question_type`**, **`question_focus`**, etc. – Original metadata fields included as context.

---

### Example

```
Question:
How does rivastigmine interact with over-the-counter sleep medication?

Reference Answer:
Tell your doctor and pharmacist what prescription and nonprescription medications,
vitamins, nutritional supplements, and herbal products you are taking or plan to take.
Be sure to mention any of the following: antihistamines; aspirin and other NSAIDs such
as ibuprofen and naproxen; bethanechol; ipratropium; and medications for Alzheimer’s disease,
glaucoma, irritable bowel disease, motion sickness, ulcers, or urinary problems.
Your doctor may need to change the doses of your medications or monitor you carefully for side effects.
```

---

### References

**MedicationQA Dataset**
```bibtex
@inproceedings{BenAbacha:MEDINFO19, 
	   author    = {Asma {Ben Abacha} and Yassine Mrabet and Mark Sharp and
   Travis Goodwin and Sonya E. Shooshan and Dina Demner{-}Fushman},    
	   title     = {Bridging the Gap between Consumers’ Medication Questions and Trusted Answers}, 
	   booktitle = {MEDINFO 2019},   
	   year      = {2019}, 
   abstract = {This paper addresses the task of answering consumer health questions about medications. To better understand the challenge and needs in terms of methods and resources, we first introduce a gold standard corpus for Medication Question Answering created using real consumer questions. The gold standard consists of six hundred and seventy-four question-answer pairs with annotations of the question focus and type and the answer source. We first present the manual annotation and answering process. In the second part of this paper, we test the performance of recurrent and convolutional neural networks in question type identification and focus recognition. Finally, we discuss the research insights from both the dataset creation process and our experiments. This study provides new resources and experiments on answering consumers’ medication questions and discusses the limitations and directions for future research efforts.}}  
}
```
