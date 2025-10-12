# medexqa-env- by mnishant2

### Overview
- **Environment ID**: `medexqa`
- **Short description**: Medical QA with multiple-choice questions and explanations across five underrepresented medical specialties
- **Tags**: medical, clinical, single-turn, multiple-choice, explanations, train, evaluation

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
- **System Prompt**: Uses the authors' prompt from their evaluation code:
  ```
  "The following is a multiple-choice question. Please choose the most suitable one
  among A, B, C and D as the answer to this question. Your answer should be paired
  with an explanation why you chose that answer."
  ```
- **Parser**: `Parser` or `ThinkParser`, with `extract_fn=extract_boxed_answer` for strict letter-in-\boxed{}-format parsing
- **Rubric overview**:
  - MCQ-only mode: Binary scoring based on correctly boxed letter choice
  - Full evaluation mode: Weighted combination of MCQ accuracy + explanation quality (using LLM-as-judge)

### Quickstart

Run MCQ-only evaluation (default):

```bash
uv run vf-eval medexqa -m gpt-4.1-mini
```

Run with explanation evaluation:

```bash
export JUDGE_API_KEY=sk-...
uv run vf-eval medexqa -m gpt-4.1-mini -a '{"use_explanations": true}'
```

Configure model and sampling:

```bash
uv run vf-eval medexqa \
    -m gpt-4.1-mini   \
    -n -1 -r 3 -t 1024 -T 0.7  \
    -a '{"use_think": false, "use_explanations": false}'
```

### Environment Arguments

| Arg                  | Type  | Default       | Description                                                                        |
| -------------------- | ----- | ------------- | ---------------------------------------------------------------------------------- |
| `use_think`          | bool  | `False`       | Whether to check for `<think>...</think>` formatting with `ThinkParser`           |
| `use_explanations`   | bool  | `False`       | Whether to evaluate explanation quality using LLM-as-judge                         |
| `mcq_weight`         | float | `0.5`         | Weight for MCQ accuracy (only used when `use_explanations=True`)                  |
| `explanation_weight` | float | `0.5`         | Weight for explanation quality (only used when `use_explanations=True`)           |
| `judge_model`        | str   | `gpt-4o-mini` | Model to use for judging explanations                                              |
| `judge_base_url`     | str   | `None`        | Base URL for judge model API                                                       |
| `judge_api_key`      | str   | `None`        | API key for judge (falls back to `JUDGE_API_KEY` or `OPENAI_API_KEY` env vars)    |

### Metrics

**MCQ-Only Mode** (`use_explanations=False`):

| Metric | Weight | Meaning |
| ------ | ------ | ------- |
| `correct_answer_reward_func` | 1.0 | 1.0 if parsed letter is correct, else 0.0 |
| `parser.get_format_reward_func()` | 0.0 | Optional format adherence (not counted) |

**Full Evaluation Mode** (`use_explanations=True`):

| Metric | Weight (default) | Meaning |
| ------ | ---------------- | ------- |
| `correct_answer_reward_func` | 0.5 | 1.0 if parsed letter is correct, else 0.0 |
| `explanation_quality_reward` | 0.5 | 0.0-1.0 score from LLM judge comparing model's explanation to two reference explanations |

**Explanation Judge Criteria:**
- Medical accuracy
- Relevance to the question
- Clarity and completeness
- Proper use of medical concepts

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
uv run vf-eval medexqa -m gpt-4.1-mini -n 5
```

#### 3. Full MCQ Evaluation
```bash
export OPENAI_API_KEY=sk-...
uv run vf-eval medexqa -m gpt-4.1-mini -n -1 -s
```

#### 4. With Explanation Evaluation
```bash
export JUDGE_API_KEY=sk-...
uv run vf-eval medexqa -m gpt-4.1-mini -n -1 -a '{"use_explanations": true}' -s
```

#### 5. With Think Tags
```bash
uv run vf-eval medexqa -m gpt-4.1-mini -n -1 -a '{"use_think": true}'
```

### Citation

```bibtex
@article{kim2024medexqa,
  title={MedExQA: Medical Question Answering Benchmark with Multiple Explanations},
  author={Kim, Yunsoo and Wu, Jinge and Abdulle, Yusuf and Wu, Honghan},
  journal={arXiv preprint arXiv:2406.06331},
  year={2024}
}
```
