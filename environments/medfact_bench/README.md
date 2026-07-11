# MedFact-Bench

## Overview

- **Environment ID**: `medfact_bench`
- **Short description**: Zero-shot medical claim verification across five benchmark components.
- **Tags**: medical, fact-verification, classification, single-turn, evaluation

`medfact_bench` implements the evaluation protocol introduced in [Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution](https://arxiv.org/abs/2603.05308). The environment is evaluation-only and does not expose a training dataset or training reward.

## Datasets

- **Primary dataset**: [`ncbi/MedFact-Bench`](https://huggingface.co/datasets/ncbi/MedFact-Bench)
- **Dataset revision**: `249028caf7ad5a3e63331269a606f4b2696693ed`
- **Paper**: [Med-V1](https://arxiv.org/abs/2603.05308)

| Component | Examples |
| --- | ---: |
| SciFact | 340 |
| HealthVer | 903 |
| MedAESQA | 9,106 |
| PubMedQA-Fact | 500 |
| BioASQ-Fact | 3,425 |
| **Total** | **14,274** |

The loader also accepts a local Parquet copy. Local loading preserves the original row order and duplicate rows while validating the required schema, component names, labels, non-null fields, and global system prompt.

## Task

- **Type**: single-turn, three-class classification
- **Labels**: `SUPPORT`, `NEI`, and `CONTRADICT`
- **Rubric overview**: exact label accuracy with separate parsing and format diagnostics

Each example contains a medical claim and a source article. The dataset-provided Med-V1 system and user prompts are used unchanged. The model assigns a five-point score that maps to the benchmark labels:

| Score | Prediction |
| ---: | --- |
| `1`, `2` | `SUPPORT` |
| `0` | `NEI` |
| `-1`, `-2` | `CONTRADICT` |

The parser reads the first `<score>...</score>` tag, trims whitespace, and accepts only integer values from `-2` through `2`. A valid `<think>` block is not required to classify a completion. Missing, malformed, or out-of-range scores are labeled `INVALID`, receive zero accuracy, and remain in all reporting denominators.

### Prompt Format

The dataset provides one shared system prompt, reproduced below. The environment passes it to the model unchanged:

```text
You are a fact-checking expert trained in evidence-based medicine. Your task is to evaluate how strongly an *article* agrees or disagrees with a *claim*. The *article* is retrieved from a search engine using the *claim* as the query.

Use the following five-point scale:
   - **-2 Strong Contradiction**  – The article clearly and directly refutes the claim.
   - **-1 Partial Contradiction** – The article provides mixed or indirect evidence against the claim.
   - ** 0 Neutral / Unrelated**   – The article does not address the claim, offers insufficient information, or is irrelevant to the claim.
   - ** 1 Partial Agreement**	 – The article offers some indirect or tentative support for the claim.
   - ** 2 Strong Agreement**	 – The article explicitly and strongly supports the claim.

Note that the *article* might not describe the exact same subjects, interventions, or measurements as the *claim*. In this case, please note the difference and assign a score of 0.

Output in two parts only and do not output anything else:
<think>[your detailed, step‐by‐step explanation for scoring]</think>
<score>[the integer score only, i.e., -2, -1, 0, 1, or 2]</score>
```

The square-bracketed phrases describe placeholder content; the brackets themselves are not required in model completions.

Each dataset row provides a user prompt with this structure:

```text
Article:
{article}

Claim:
{claim}
```

## Quickstart

Run a small evaluation with Prime:

```bash
prime eval run medfact_bench -m "openai/gpt-5-mini" -n 5 -r 1
```

Run the complete benchmark with MedMarks:

```bash
medarc-eval medfact_bench \
  -m "openai/gpt-5-mini" \
  -n -1 \
  -r 1 \
  --subset all
```

Use `--dataset-path /absolute/path/to/MedFact-Bench.parquet` to evaluate from a local dataset copy.

## Environment Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `subset` | `str` or `MedFactSubset` | `all` | Component to evaluate: `all`, `scifact`, `healthver`, `medaesqa`, `pubmedqa-fact`, or `bioasq-fact`. |
| `dataset_path` | `str` or `None` | `None` | Optional local Parquet path. Local data takes precedence over Hugging Face. |
| `cache_dir` | `str` or `None` | `None` | Optional Hugging Face datasets cache directory. |

## Metrics

| Metric | Weight | Meaning |
| --- | ---: | --- |
| `accuracy` | 1.0 | Exact three-class accuracy. This is also the Verifiers reward and `avg_reward`. |
| `parseable_score` | 0.0 | Whether a valid five-point score was parsed. |
| `strict_format` | 0.0 | Whether the completion exactly follows the requested think-and-score block format. |

`avg_reward` is example-weighted micro-accuracy. The paper's primary benchmark metric is macro-accuracy across the five components and is produced by the reporting command below.

## Reporting

Generate benchmark-specific metrics from the raw `results.jsonl` before running standard MedMarks result processing:

```bash
python -m medfact_bench.reporting RESULTS_JSONL \
  --output MEDFACT_REPORT_JSON \
  --metadata METADATA_JSON
```

Without explicit output paths, the command writes `medfact_report.json` beside `results.jsonl` and updates the sibling `metadata.json` non-destructively. Use `--no-update-metadata` to leave metadata unchanged.

The report includes valid and invalid prediction counts, parse and strict-format rates, micro-accuracy, macro-accuracy over observed components, per-component accuracy, per-class precision/recall/F1, macro-F1, a confusion matrix with `INVALID`, and benchmark coverage. `paper_macro_accuracy` is populated only when all 14,274 examples have exact component coverage; partial runs set it to `null` and set `paper_comparable` to `false`.

## Published Results

Figure 3 of the Med-V1 paper reports the following zero-shot accuracies. The small models are the locally runnable 3B base checkpoints. The frontier models were evaluated through Azure-hosted APIs. `Med-V1-L3B` and `Med-V1-Q3B` are fine-tuned Med-V1 models trained on MedFact-Synth, then evaluated zero-shot on MedFact-Bench.

### Small Models

| Model | SciFact | HealthVer | MedAESQA | PubMedQA-Fact | BioASQ-Fact | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct | .600 | .435 | .437 | .550 | .531 | .511 |
| Qwen2.5-3B-Instruct | .638 | .447 | .577 | .470 | .444 | .515 |

### Frontier Models

| Model | SciFact | HealthVer | MedAESQA | PubMedQA-Fact | BioASQ-Fact | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.3-70B-Instruct | .812 | .597 | .687 | .774 | .771 | .728 |
| o3-mini | .832 | .591 | .733 | .778 | .747 | .736 |
| GPT-4o-mini | .847 | .580 | .704 | .734 | .720 | .717 |
| GPT-4o | .832 | .576 | .717 | .776 | .777 | .736 |
| GPT-5 | .818 | .615 | .703 | .788 | .753 | .735 |

### Fine-Tuned Models

| Model | SciFact | HealthVer | MedAESQA | PubMedQA-Fact | BioASQ-Fact | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Med-V1-L3B | .844 | .575 | .748 | .746 | .725 | .728 |
| Med-V1-Q3B | .856 | .588 | .733 | .764 | .717 | .732 |

Macro is the unweighted mean of the five component accuracies and is the paper's primary metric. These values are reference results from the publication. Evaluation artifacts should preserve the dataset revision, model revision, prompts, sampling configuration, and raw predictions needed to interpret differences between runs.

## Authors

This environment has been put together by:

Kouate Muhamed - ([@KouateMuhamed](https://github.com/KouateMuhamed))
