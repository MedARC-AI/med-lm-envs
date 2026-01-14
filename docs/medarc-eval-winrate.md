# Computing Win Rates

Compare model performance across benchmarks using HELM-style win rate calculations. Win rates measure how often one model outperforms another on the same questions.

## Quick Start

```bash
# Compute win rates from processed results
medarc-eval winrate

# See which models are available
medarc-eval winrate --list-models

# Specify directories
medarc-eval winrate --processed-dir runs/processed --output-dir runs/winrate
```

## Prerequisites

Win rate computation requires processed parquet files with an `env_index.json`:

```bash
# If you haven't processed yet:
medarc-eval process
```

## How Win Rates Work

For each pair of models (A, B) on each benchmark:
1. Find questions both models answered
2. Compare scores on each question
3. Count: A wins, B wins, ties
4. Win rate = (A wins + 0.5 × ties) / total

The final win rate aggregates across all benchmarks using configurable weighting.

## Output Files

```
runs/winrate/
├── winrates-2026-01-14T12-00-00.json    # Timestamped results
├── winrates-2026-01-14T12-00-00.csv     # Spreadsheet-friendly
├── latest.json                           # Always points to newest
└── latest.csv
```

### Output Format

The JSON output includes:
- Per-model aggregate win rates
- Pairwise comparison matrices
- Per-benchmark breakdowns
- Computation metadata

## Common Options

### Model Selection

| Flag | Description |
|------|-------------|
| `--list-models` | Show available models and exit |
| `--include-model MODEL` | Only include specified models (repeatable) |
| `--exclude-model MODEL` | Exclude specified models (repeatable) |

### Win Rate Calculation

| Flag | Description | Default |
|------|-------------|---------|
| `--missing-policy` | How to handle missing scores: `zero` or `neg-inf` | `zero` |
| `--epsilon` | Tie tolerance (scores within epsilon are ties) | `1e-9` |
| `--min-common` | Minimum shared examples for valid comparison | `0` |

### Benchmark Weighting

| Flag | Description | Default |
|------|-------------|---------|
| `--weight-policy` | How to weight benchmarks: `equal`, `ln`, `sqrt`, `cap` | `equal` |
| `--weight-cap` | Maximum weight per benchmark (for `cap` policy) | `0` |

### Partial Data Handling

| Flag | Description |
|------|-------------|
| `--partial-datasets strict` | Only use benchmarks where all selected models have data |
| `--partial-datasets include` | Include benchmarks with partial coverage |

## Using a Config File

```yaml
# winrate-config.yaml
processed_dir: runs/processed
output_dir: runs/winrate

# Calculation settings
missing_policy: neg-inf
epsilon: 1.0e-9
min_common: 10
weight_policy: ln

# Model filtering
exclude_model:
  - baseline-model
  - deprecated-v1
```

```bash
medarc-eval winrate --config winrate-config.yaml
```

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

Only use benchmarks where all models have results:

```bash
medarc-eval winrate \
  --include-model gpt-4o \
  --include-model gpt-4o-mini \
  --partial-datasets strict
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
  --hf-processed-repo your-org/processed-benchmarks \
  --hf-processed-pull \
  --hf-token $HF_TOKEN
```

### Upload Win Rates to Hub

```bash
medarc-eval winrate \
  --hf-winrate-repo your-org/winrate-results \
  --hf-token $HF_TOKEN \
  --hf-private
```

### Full Config with HF

```yaml
# winrate-config.yaml
processed_dir: runs/processed
output_dir: runs/winrate

missing_policy: neg-inf
weight_policy: ln

hf:
  repo: your-org/processed-data          # Pull processed from here
  winrate_repo: your-org/winrate-results # Upload results here
  branch: main
  token: ${HF_TOKEN}
  private: true
```

## Interpreting Results

### Win Rate Table (CSV)

| model | win_rate | vs_gpt-4o | vs_gpt-4o-mini | vs_claude |
|-------|----------|-----------|----------------|-----------|
| gpt-4o | 0.72 | - | 0.85 | 0.58 |
| gpt-4o-mini | 0.45 | 0.15 | - | 0.32 |
| claude-3-5-sonnet | 0.68 | 0.42 | 0.68 | - |

- **win_rate**: Aggregate win rate across all models
- **vs_X columns**: Pairwise win rate against model X
- Values > 0.5 mean the row model wins more often

### JSON Structure

```json
{
  "models": ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
  "aggregate_winrates": {
    "gpt-4o": 0.72,
    "gpt-4o-mini": 0.45,
    "claude-3-5-sonnet": 0.68
  },
  "pairwise": {
    "gpt-4o": {
      "gpt-4o-mini": {"win_rate": 0.85, "wins": 850, "losses": 150, "ties": 0},
      "claude-3-5-sonnet": {"win_rate": 0.58, ...}
    }
  },
  "per_benchmark": { ... }
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
