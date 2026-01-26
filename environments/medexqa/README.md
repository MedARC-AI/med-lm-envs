# medexqa-env- by mnishant2

### Overview
- **Environment ID**: `medexqa`
- **Short description**: Medical QA with multiple-choice questions and explanations across five underrepresented medical specialties
- **Tags**: medical, clinical, single-turn, multiple-choice, explanations, evaluation

### Datasets
- **Primary dataset(s)**: MedExQA
- **Source links**: [Paper](https://arxiv.org/abs/2406.06331), [HuggingFace Dataset](https://huggingface.co/datasets/bluesky333/MedExQA), [GitHub](https://github.com/knowlab/MedExQA)
- **Split sizes**:

    | Specialty                   | Dev | Test | Total |
    | --------------------------- | --- | ---- | ----- |
    | Biomedical Engineering      | 4   | 144  | 148   |
    | Clinical Laboratory Science | 9   | 368  | 377   |
    | Clinical Psychology         | 3   | 108  | 111   |
    | Occupational Therapy        | 5   | 189  | 194   |
    | Speech Language Pathology   | 4   | 131  | 135   |
    | **Total**                   | **25** | **940** | **965** |

### Task
- **Type**: single-turn
- **Prompting**: Uses the authors' instruction embedded in the user message; options A/B/C/D are included.
  ```
  The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question. Your answer should be paired with an explanation why you chose that answer.
  ```
- **Answer extraction [authors' logic](https://github.com/knowlab/MedExQA/blob/9a5b34af103b0c8ba0c00906e278f6572249fafa/evaluate_pipe_MedExQA.py)** :
  - Canonical letter extraction using a sequence of regex patterns (e.g., explicit "Answer is A:", leading letter, etc.)
  - If no explicit letter is found, fuzzy matching (thefuzz) maps the generated text to the closest option and returns the corresponding letter
- **Parser**: `Parser` or `ThinkParser` with `extract_fn=extract_boxed_answer` supported for think-mode; MCQ scoring uses the authors' extraction logic above.
- Run Evaluation per specialty or on multiple specialties
- Use lexical metrics('rougeL', 'bleu', 'bertscore', 'meteor') or use an LLM-as-a-judge for explanation evaluation
- **Rubric overview**:
  - MCQ accuracy: 0 or 100 per example
  - Explanation score: 0–100 per example (lexical metrics average); 0 if the answer is wrong
  - Combined score: weighted average of MCQ and explanation (`mcq_weight`, `explanation_weight`)
- **Model Download**:
  In the first run it will download `wordnet`, `NLTK` and `sciBERT` models for running the lexical metrics

### Quickstart

- Run MCQ-only (no explanation scoring):
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -a '{"use_explanations": false}'
```

- Run with explanation scoring (lexical metrics):
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -a '{"use_explanations": true}'
```

- Use LLM-as-judge for explanations (instead of lexical metrics):
```bash
export JUDGE_API_KEY=sk-...
uv run vf-eval medexqa -m gpt-4.1-mini -a '{"use_explanations": true, "use_judge": true, "judge_model": "gpt-4o-mini"}'
```

- Configure sampling and rollouts:
```bash
uv run vf-eval medexqa \
  -m gpt-4.1-mini \
  -n -1 -r 3 -t 1024 -T 0.7 \
  -a '{"use_think": false, "use_explanations": true, "mcq_weight": 0.5, "explanation_weight": 0.5}'
```

### Environment Arguments

| Arg                    | Type                   | Default        | Description |
| ---------------------- | ---------------------- | -------------- | ----------- |
| `specialty`            | list[str] \/ str \| None | `None`         | Select one or more specialties. Codes: `BE`, `CLS`, `CP`, `OT`, `SLP`. `None`\/`ALL` loads all. |
| `use_think`            | bool                   | `False`        | Use `ThinkParser` to support `<think>...</think>` blocks. |
| `use_explanations`     | bool                   | `True`         | Whether to compute explanation scores. |
| `explanation_metrics`  | list[str] \/ str \| None | `None`         | Lexical metrics to use: any of `rougeL`, `bleu`, `meteor`, `bertscore`. `None`\/`"all"` averages all four. |
| `mcq_weight`           | float                  | `0.5`          | Weight for MCQ accuracy in the combined score. |
| `explanation_weight`   | float                  | `0.5`          | Weight for explanation in the combined score. |
| `use_judge`            | bool                   | `False`        | Use LLM-as-judge for explanations instead of lexical metrics. |
| `judge_model`          | str                    | `gpt-4o-mini`  | Judge model name. |
| `judge_base_url`       | str \| None            | `None`         | Judge API base URL. |
| `judge_api_key`        | str \| None            | `None`         | Judge API key (falls back to `JUDGE_API_KEY` or `OPENAI_API_KEY`). |
| `seed`                 | int \| None            | `None`         | When multiple specialties are selected, shuffles the combined eval set with this seed. |

### Metrics

- **Answer accuracy (per example)**: 0 or 100. Uses authors' regex+fuzzy logic to extract a letter.
- **Explanation score (per example)**: 0–100. If the answer is wrong, the explanation score is 0.
  - Lexical metrics supported: `rougeL`, `bleu`, `meteor`, `bertscore` (w/ SciBERT `allenai/scibert_scivocab_uncased`).
  - Selection via `explanation_metrics` (list or `'all'`/`None` to average all four).
- **Combined score**: `mcq_weight * accuracy + explanation_weight * explanation`.

Optional LLM-as-judge for explanations:
- Set `use_explanations=true` and `use_judge=true` to replace lexical metrics with judge scoring (0–100 after scaling).
- Criteria include medical accuracy, relevance, clarity, completeness, and use of medical concepts. 0 if the answer from string matching is wrong.

### Specialty Selection and Macro Average

- Single specialty by code:
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -a '{"specialty": "CLS"}'
```

- Multiple specialties:
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -a '{"specialty": ["CLS", "CP"], "seed": 42}'
```

- All specialties:
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -a '{"specialty": "ALL"}'
```

## IMPORTANT: Macro-average accuracy (as reported in the paper):
- Run each specialty separately and average the per-run average answer accuracies; or
- Run multiple specialties with `-s` to save results. Each saved example includes its `specialty` in `info`, along with the `per-example answer_accuracy_reward`. Use the saved JSONL to compute per-specialty accuracies and then take the unweighted mean across specialties.

### Testing Instructions

#### 1. Environment Setup
```bash
# Navigate to repository root
cd /data/storage_hpc_nishant/med-lm-envs

# Sync uv environment
uv sync
```

#### 2. Quick Validation Test (MCQ-only)
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -n 5 -a '{"use_explanations": false}'
```

#### 3. Full Evaluation with Save
```bash
export OPENAI_API_KEY=sk-...
uv run vf-eval medexqa -m gpt-4.1-mini -n -1 -s -a '{"specialty": "ALL", "use_explanations": true}'
```

#### 4. LLM-as-Judge for Explanations
```bash
export JUDGE_API_KEY=sk-...
uv run vf-eval medexqa -m gpt-4.1-mini -n -1 -s -a '{"use_explanations": true, "use_judge": true, "mcq_weight": 0.5, "explanation_weight": 0.5}'
```

#### 5. With Think Tags
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -n -1 -a '{"use_think": true}'
```

#### 6. Example Run with openrouter 
```bash
export OPENROUTER_API_KEY=....
uv run vf-eval medexqa  -m openai/gpt-oss-20b:free -b https://openrouter.ai/api/v1 -k OPENAI_API_KEY -n 10 -r 1 -c 1 -a '{"use_explanations": true, "explanation_metrics": "all", "specialty": ["BE", "OT"]}' -s
```
output 
```bash
Rewards:
reward: avg - 59.416, std - 19.928
r1: [67.79, 65.809, 64.158, 66.619, 69.124, 0.0, 66.957, 66.327, 66.87, 60.503]
answer_accuracy_reward: avg - 90.000, std - 30.000
r1: [100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 100.0]
explanation_reward: avg - 28.832, std - 10.577
r1: [35.58, 31.618, 28.316, 33.239, 38.249, 0.0, 33.915, 32.653, 33.741, 21.006]
```
### Authors
This environment has been put together by:

Nishant Mishra - ([mnishant2](https://github.com/mnishant2))

### Citation

```bibtex
@article{kim2024medexqa,
  title={MedExQA: Medical Question Answering Benchmark with Multiple Explanations},
  author={Kim, Yunsoo and Wu, Jinge and Abdulle, Yusuf and Wu, Honghan},
  journal={arXiv preprint arXiv:2406.06331},
  year={2024}
}
```
