# Computing Win Rates

Compare model performance across benchmarks using HELM-style win rate calculations. Win rates measure how often one model outperforms another on the same questions.

## Quick Start

```bash
# Compute win rates from processed results
medarc-eval winrate

# See which models are available
medarc-eval winrate --list-models

# Specify directories
medarc-eval winrate --processed-dir runs/processed --output-dir runs/processed/winrate
```

## Prerequisites

Win rate computation requires processed parquet files with an `env_index.json`:

```bash
# If you haven't processed yet:
medarc-eval process --runs-dir runs/evals
```

## How Win Rates Work

For each pair of models (A, B) on each benchmark:
1. Average rollouts per `(example_id, model_id)`
2. Compare questions where at least one model has a reward
3. If one side is missing, fill it according to `--missing-policy` (`neg-inf` or `zero`)
4. Count: A wins, B wins, ties
5. Win rate = (A wins + 0.5 × ties) / total used questions

The final win rate aggregates across all benchmarks using configurable weighting.

Winrate also emits a missingness summary so partial dataset coverage is visible. The report counts missing
`(dataset, model)` pairs after rollout averaging, including both absent rows and null reward values.

## Output Files

```
runs/processed/winrate/
├── winrates-20260114T120000Z.json       # Timestamped results
├── winrates-20260114T120000Z.csv        # Spreadsheet-friendly
├── latest.json                           # Always points to newest
└── latest.csv
```

If you pass `--output /path/to/file.json`, winrate writes only that JSON file and skips `latest.json` plus all CSV outputs.

### Output Format

The JSON output includes:
- Per-model aggregate win rates
- Per-opponent `vs` breakdowns
- Per-dataset average rewards and question counts

## Common Options

### Model Selection

| Flag | Description |
|------|-------------|
| `--list-models` | Show available models and exit |
| `--include-model MODEL` | Only include specified models (repeatable) |
| `--exclude-model MODEL` | Exclude specified models (repeatable) |
| `--exclude-dataset DATASET` | Exclude specified datasets/env ids (repeatable) |

### Win Rate Calculation

| Flag | Description | Default |
|------|-------------|---------|
| `--missing-policy` | How to handle missing scores: `zero` or `neg-inf` | `neg-inf` |
| `--epsilon` | Tie tolerance (scores within epsilon are ties) | `1e-9` |
| `--min-common` | Minimum shared examples for valid comparison | `0` |

### Benchmark Weighting

| Flag | Description | Default |
|------|-------------|---------|
| `--weight-policy` | How to weight benchmarks: `equal`, `ln`, `sqrt`, `cap` | `ln` |
| `--weight-cap` | Maximum weight per benchmark (for `cap` policy) | `0` |

### Dataset Coverage

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-coverage all-models` | Enforce intersection of datasets across the compared models | `all-models` |
| `--dataset-coverage per-model` | Legacy behavior (each model may be averaged over different datasets) | |

### Partial Data Handling

| Flag | Description |
|------|-------------|
| `--partial-datasets strict` | When `--include-model` is set, drop datasets missing any included model |
| `--partial-datasets include` | When `--include-model` is set, keep datasets and treat missing models as all-missing |

`--partial-datasets include` is usually paired with `--dataset-coverage per-model`. With the default `all-models` coverage, datasets missing any required model are still dropped later.

## Using a Config File

```yaml
# process-config.yaml
runs_dir: runs/evals

process:
  dir: processed

winrate:
  dir: winrate
  missing_policy: neg-inf
  epsilon: 1.0e-9
  min_common: 10
  weight_policy: ln
  exclude_model:
    - baseline-model
    - deprecated-v1
  exclude_datasets:
    - med_dialog
```

```bash
medarc-eval winrate --config process-config.yaml
```

Supported config schema for `medarc-eval winrate`:

- Top-level `process:` can provide `dir` or `output_dir`; this becomes the default `processed_dir`.
- Top-level `winrate:` provides winrate-specific defaults.
- Top-level `hf:` provides shared HF settings. Use `hf.winrate_dir` to control where winrate artifacts upload inside the repo.

## Example Workflows

### Compare Specific Models

```bash
# Only compare these two models
medarc-eval winrate \
  --include-model gpt-4o \
  --include-model claude-3-5-sonnet
```

### Exclude Baseline Models

```bash
medarc-eval winrate --exclude-model random-baseline
```

### Strict Benchmark Coverage

Only use datasets where all compared models have results:

```bash
medarc-eval winrate \
  --include-model gpt-4o \
  --include-model gpt-4o-mini \
  --dataset-coverage all-models
```

### Custom Weighting

Weight benchmarks by log of dataset size (larger benchmarks count more):

```bash
medarc-eval winrate --weight-policy ln
```

## Hugging Face Integration

### Pull Processed Data from Hub

```bash
medarc-eval winrate \
  --hf-repo your-org/processed-benchmarks \
  --hf-processed-pull \
  --hf-token $HF_TOKEN
```

### Upload Win Rates to Hub

```bash
medarc-eval winrate \
  --hf-repo your-org/processed-benchmarks \
  --hf-winrate-dir winrate \
  --hf-token $HF_TOKEN \
  --hf-private
```

### Full Config with HF

```yaml
# process-config.yaml
runs_dir: runs/evals

process:
  dir: processed

winrate:
  dir: winrate
  missing_policy: neg-inf
  weight_policy: ln

hf:
  repo: your-org/processed-data # Pull processed from here; upload winrate here
  winrate_dir: winrate          # Subdirectory in repo for winrate artifacts (default: winrate)
  branch: main
  token: ${HF_TOKEN}
  private: true
```

`hf.token` accepts either a literal token string or an environment reference like `$HF_TOKEN` / `${HF_TOKEN}`.

`hf.winrate_dir` and `--hf-winrate-dir` both set the path inside the HF repo where `latest.json`, `latest.csv`, and timestamped winrate outputs are uploaded.

## Interpreting Results

### Win Rate Table (CSV)

| model | weighted_winrate | simple_winrate | medqa | pubmedqa | num_datasets |
|-------|------------------|----------------|-------|-----------|--------------|
| gpt-4o | 0.72 | 0.70 | 0.84 | 0.77 | 2 |
| gpt-4o-mini | 0.45 | 0.43 | 0.61 | 0.39 | 2 |

- **weighted_winrate** / **simple_winrate**: Aggregate mean winrate across retained datasets
- Dataset columns: Average reward on that dataset, not pairwise winrate columns
- `num_datasets`: Number of datasets retained for that model after filtering/coverage rules

### JSON Structure

```json
{
  "models": {
    "gpt-4o": {
      "mean_winrate": {
        "simple_mean": 0.72,
        "weighted_mean": 0.74,
        "n_datasets": 2
      },
      "vs": {
        "gpt-4o-mini": {
          "mean_winrate": {
            "simple_mean": 0.85,
            "weighted_mean": 0.84
          },
          "per_dataset": {
            "medqa": 0.90,
            "pubmedqa": 0.80
          },
          "n_datasets": 2
        }
      },
      "avg_reward_per_dataset": {
        "medqa": 0.84,
        "pubmedqa": 0.77
      }
    }
  },
  "datasets": {
    "medqa": {
      "avg_reward_per_model": {
        "gpt-4o": 0.84,
        "gpt-4o-mini": 0.61
      },
      "n_questions": 1273
    }
  },
}
```

## Troubleshooting

### "No models found"

- Ensure `medarc-eval process` has been run
- Check that `env_index.json` exists in `--processed-dir`

### Unexpected win rates

- Check `--min-common` isn't filtering out comparisons
- Review `--missing-policy` (use `neg-inf` to penalize missing answers)
- Verify models were evaluated on the same benchmark variants
- If using `--partial-datasets include`, also consider `--dataset-coverage per-model`
