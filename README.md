# Medmarks

[![Website](https://img.shields.io/badge/website-medmarks.ai-0f766e)](https://medmarks.ai)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01417-b31b1b.svg)](https://arxiv.org/abs/2605.01417)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Open-source LLM benchmark suite for medical tasks.

[medmarks.ai](https://medmarks.ai) | [arXiv:2605.01417](https://arxiv.org/abs/2605.01417)

Medmarks is a comprehensive benchmark suite for evaluating medical capabilities in large language models. It includes 30 open-source benchmarks spanning question answering, information extraction, consumer health questions, clinical reasoning, EHR interactions, medical calculations, and open-ended medical tasks.

This repository contains the runnable benchmark environments, evaluation configs, result processing tools, and win-rate analysis pipeline used for Medmarks. It also contains the [`medarc_verifiers` Python library](docs/README.md), which provides the shared CLI, parsers, rewards, judging utilities, and orchestration helpers used by the benchmark environments.

## Benchmark Suite

Medmarks is organized into three practical subsets:

| Subset | Description |
|--------|-------------|
| Medmarks-V | Verifiable tasks, including multiple-choice QA and other tasks with deterministic or programmatic grading |
| Medmarks-OE | Open-ended tasks evaluated with LLM-as-a-Judge |
| Medmarks-T | Experimental training-capable environments with train/test splits for post-training and RL experiments |

The benchmark suite is implemented as [verifiers](https://github.com/primeintellect-ai/verifiers) environments under [`environments/`](environments/). The main runnable suite configs are:

| Config | Purpose |
|--------|---------|
| [`configs/eval/medmarks-verified.toml`](configs/eval/medmarks-verified.toml) | Medmarks-V suite |
| [`configs/eval/medmarks-open_ended.toml`](configs/eval/medmarks-open_ended.toml) | Medmarks-OE suite |
| [`configs/eval/smoke.toml`](configs/eval/smoke.toml) | Small sanity-check run |

## Quick Start

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Run a single benchmark:

```bash
uv run medarc-eval medqa -m openai/gpt-4.1-mini -n 25
```

Run a Medmarks suite config:

```bash
uv run medarc-eval bench --config configs/eval/medmarks-verified.toml
```

Process outputs and compute win rates:

```bash
uv run medarc-eval process --runs-dir runs/evals
uv run medarc-eval winrate
```

Evaluation outputs are written under `runs/evals/`, processed parquet files under `runs/processed/`, and win-rate summaries under `runs/processed/winrate/`.

## Documentation

| Page | Description |
|------|-------------|
| [`docs/getting-started.md`](docs/getting-started.md) | Developer setup, environment authoring, and local workflow |
| [`docs/medarc-eval.md`](docs/medarc-eval.md) | Full `medarc-eval` CLI documentation |
| [`docs/medarc-eval-bench.md`](docs/medarc-eval-bench.md) | TOML benchmark suite execution |
| [`docs/medarc-eval-process.md`](docs/medarc-eval-process.md) | Processing eval outputs into parquet |
| [`docs/medarc-eval-winrate.md`](docs/medarc-eval-winrate.md) | HELM-style win-rate computation |
| [`docs/medarc-orchestrate.md`](docs/medarc-orchestrate.md) | Running local vLLM benchmark jobs with Docker or Slurm/Pyxis |

## Datasets

`--` indicates no dedicated training split. `Not specified` means we found no explicit dataset license in the dataset source. Evaluated counts reflect the effective Medmarks evaluation split or configured subset; MedDialog is intentionally capped at the first 2,500 examples.

| Dataset | Description | License / terms | # Evaluated | # Training |
|---------|-------------|-----------------|------------:|-----------:|
| **Medmarks-V (Verifiable)** | | | | |
| CareQA | Healthcare QA exam questions with multiple-choice reasoning questions, English subset. | Apache-2.0 | 5,621 | -- |
| HEAD-QA v2 | Extended healthcare questions spanning 10 years of Spanish professional exams, English subset. | MIT | 12,751 | -- |
| LongHealth | Long-context synthetic patient cases with information extraction and sorting tasks, task1 and task2 splits. | Apache-2.0 | 1,200 | -- |
| M-ARC | Long-tail medical questions designed to test model resistance to inflexible clinical reasoning patterns. | Apache-2.0 | 100 | -- |
| Med-HALT | Clinical Reasoning Hallucination detection via false confidence tests and "none of the above" recognition. | Apache-2.0 | 22,152 | -- |
| MedCalc-Bench | Clinical calculator questions evaluating medical computation and formula application skills. | CC-BY-SA-4.0 | 1,100 | 10,538 |
| MedConceptsQA | Multiple-choice questions on medical coding systems, e.g., ICD-9, ICD-10, etc., only ICD-10CM subsamples evaluated. | Not specified | 6,000 | -- |
| Medbullets | USMLE Step 2 and Step 3 style clinical reasoning questions sourced from social media. | Not specified | 308 | -- |
| MedHallu | Medical hallucination detection benchmark with four domain-specific error categories derived from the PubMedQA dataset. | MIT | 2,000 | -- |
| MedMCQA | Multiple-choice questions from Indian medical entrance exams across 21 medical subjects. | Apache-2.0 | 4,183 | 182,822 |
| MedQA | Multiple-choice questions from USMLE medical licensing exams. | CC-BY-4.0 | 1,273 | 10,178 |
| MedXpertQA | High-difficulty MCQ questions with ~10 options across 17 specialties to evaluate expert-level medical knowledge, text subset. | MIT | 2,450 | -- |
| MetaMedQA | Questions testing model's awareness and recognition of unanswerable medical queries using uncertainty options. | CC-BY-4.0 | 1,373 | -- |
| MMLU-Pro-Health | Health subset of MMLU-Pro benchmark featuring general health-related questions with up to 10 answer options per question. | MIT | 818 | -- |
| PubHealthBench | Multiple-choice questions derived from UK government public health guidance documents, reviewed subset. | CC-BY-4.0 | 760 | -- |
| PubMedQA | Yes/no/maybe question answering requiring reasoning over biomedical research abstracts, labeled subset. | MIT | 500 | 211,269 |
| SCTPublic | Script Concordance Tests evaluating clinical reasoning under diagnostic uncertainty. | MIT | 174 | -- |
| SuperGPQA-Med | Graduate-level questions spanning 6 medical fields, easy and hard difficulty subsets. | ODC-BY | 1,126 | -- |
| **Medmarks-OE (Open-Ended)** | | | | |
| ACI-Bench | Clinical dialogue transcripts paired with corresponding structured clinical notes. | CC-BY-4.0 | 210 | 114 |
| AgentClinic | Multimodal multi-agent OSCE-style clinical dialogues for interactive diagnostic reasoning evaluation. | MIT | 214 | -- |
| CareQA Open | Healthcare QA exam questions with open-ended reasoning questions, English subset. | Apache-2.0 | 2,769 | -- |
| HealthBench | Multi-turn healthcare conversations evaluated using physician-written scoring rubrics. | MIT | 5,000 | -- |
| MedAgentBench v2 | Agentic electronic health record tasks requiring FHIR API interactions. | Not specified; V1 MIT | 600 | -- |
| MedCaseReasoning | Diagnostic QA with clinician-authored reasoning traces from clinical case reports. | MIT | 500 | 13,092 |
| MedDialog | Large-scale patient-doctor conversations for medical dialogue generation and understanding; Medmarks evaluates a small subsample. | Not specified | 2,500 | 205,973 |
| MedExQA | Questions with dual expert explanations across 5 underrepresented medical specialties. | CC-BY-NC-SA-4.0 | 940 | -- |
| MedicationQA | Consumer-style medication questions with expert-validated answers from MedlinePlus. | CC-BY-4.0 | 690 | -- |
| MEDEC | Medical dataset for clinical error detection, extraction, and correction in synthetic medical notes. | CC-BY-4.0 | 597 | 2,189 |
| MedR-Bench | Clinical reasoning benchmark with step-by-step diagnostic and treatment planning traces on rare disease cases. | CC-BY-SA-4.0 | 1,453 | -- |
| MTSamples | Transcribed medical operative notes and reports evaluating models on procedural summaries and clinically appropriate treatment plans. | Not specified | 559 | -- |

## Citation

```bibtex
@misc{warner2026medmarkscomprehensiveopensourcellm,
      title={Medmarks: A Comprehensive Open-Source LLM Benchmark Suite for Medical Tasks},
      author={Benjamin Warner and Ratna Sagari Grandhi and Max Kieffer and Aymane Ouraq and Saurav Panigrahi and Geetu Ambwani and Kunal Bagga and Nikhil Khandekar and Arya Hariharan and Nishant Mishra and Manish Ram and Shamus Sim Zi Yang and Ahmed Essouaied and Adepoju Jeremiah Moyondafoluwa and Robert Scholz and Bofeng Huang and Molly Beavers and Srishti Gureja and Anish Mahishi and Sameed Khan and Maxime Griot and Hunar Batra and Jean-Benoit Delbrouck and Siddhant Bharadwaj and Ronald Clark and Ashish Vashist and Anas Zafar and Leema Krishna Murali and Harsh Deshpande and Ameen Patel and William Brown and Johannes Hagemann and Connor Lane and Paul Steven Scotti and Tanishq Mathew Abraham},
      year={2026},
      eprint={2605.01417},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.01417},
}
```

## License

Medmarks code in this repository is released under the [MIT License](LICENSE). Individual benchmark datasets may have their own licenses or terms of use; consult the corresponding dataset sources and environment documentation before redistribution or commercial use.
