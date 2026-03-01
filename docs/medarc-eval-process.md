# Processing Results

Convert raw benchmark outputs into analysis-ready parquet files. This step prepares data for win rate computation and other analyses.

## Quick Start

```bash
# Process all completed jobs (uses defaults)
medarc-eval process

# Specify directories explicitly
medarc-eval process --runs-dir runs/raw --output-dir runs/processed

# Preview what would be processed
medarc-eval process --dry-run
```

## What Processing Does

1. **Discovers** jobs in `runs/raw/` and filters by manifest status (default: `completed`)
2. **Extracts** results from each job's output files
3. **Normalizes** data into a consistent schema
4. **Writes** parquet files organized by environment and model
5. **Creates** an index (`env_index.json`) for downstream tools

### Output Structure

```
runs/processed/
├── env_index.json              # Dataset inventory for winrate/analysis
├── gpt-4o/
│   ├── medqa.parquet
│   └── pubmedqa.parquet
├── gpt-4o-mini/
│   ├── medqa.parquet
│   └── pubmedqa.parquet
└── ...
```

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--runs-dir PATH` | Directory containing raw runs | `runs/raw` |
| `--output-dir PATH` | Where to write processed files | `runs/processed` |
| `--max-workers N` | Parallel worker processes | 4 |
| `--dry-run` | Show what would be processed | - |
| `--yes` | Skip confirmation prompts | - |
| `--exclude-dataset NAME` | Skip processing specific datasets/env ids (repeatable) | - |
| `--exclude-model MODEL` | Skip processing specific model ids (repeatable) | - |

## Filtering Runs

### By Completion Status

By default, `medarc-eval process` only selects jobs whose manifest status is `completed`.

Note: successful jobs are written to `run_manifest.json` with `status: completed`.

To override that default, pass one or more explicit status filters:

```bash
medarc-eval process --status completed --status failed
```

You can also gate partially complete runs by their manifest summary totals:

```bash
# Default tolerance is 2.5 percent missing
medarc-eval process --max-run-missing-pct 2.5

# Effectively disable the gate
medarc-eval process --max-run-missing-pct 100
```

### Latest Runs Only

When multiple runs exist for the same (model, environment) pair, processing uses the latest by default.

## Clean Rebuild

Delete all processed outputs and rebuild from scratch:

```bash
# Interactive confirmation
medarc-eval process --clean

# Non-interactive (for scripts)
medarc-eval process --clean --yes
```

## Using a Config File

Store common options in a YAML file:

```yaml
# process-config.yaml
runs_dir: runs/raw

process:
  dir: processed
  max_workers: 8
  max_run_missing_pct: 2.5
  exclude_datasets:
    - med_dialog
  exclude_models:
    - deprecated-v1

winrate:
  enabled: true
  dir: winrate
```

```bash
medarc-eval process --config process-config.yaml
```

CLI flags override config values.

Supported config schema for `medarc-eval process`:

- Top-level `runs_dir`: raw run root.
- Top-level `process:`: process-specific defaults.
- Optional top-level `winrate:`: embedded post-process winrate step.
- Optional top-level `hf:`: shared HF settings. For embedded winrate uploads, use `hf.winrate_dir`.

Path shortcuts:

- `process.dir` is shorthand for `process.output_dir`, resolved relative to the parent of `runs_dir`.
- `winrate.dir` is shorthand for the embedded winrate output directory, resolved under the processed output dir.

Example:

```yaml
runs_dir: runs/raw

process:
  dir: processed
  max_workers: 8

winrate:
  dir: scorecards

hf:
  repo: your-org/medical-benchmarks
  winrate_dir: scorecards/latest
```

## Hugging Face Integration

Sync processed datasets to/from the Hugging Face Hub:

```yaml
# process-config.yaml
runs_dir: runs/raw
process:
  dir: processed

hf:
  repo: your-org/medical-benchmarks
  branch: main
  token: ${HF_TOKEN}
  private: true
```

### Pull Before Processing

```bash
# Prompt before pulling
medarc-eval process --hf-repo your-org/data --hf-pull-policy prompt

# Always pull existing data first
medarc-eval process --hf-repo your-org/data --hf-pull-policy pull

# Start fresh (ignore remote)
medarc-eval process --hf-repo your-org/data --hf-pull-policy clean
```

### Push After Processing

When `--hf-repo` is set, processed files are automatically uploaded after completion.

## Chaining with Win Rates

Process and compute win rates in one step:

```bash
medarc-eval process --config process-config.yaml
```

This runs `medarc-eval winrate` automatically after processing completes when the config contains a `winrate:` section.

## Example Workflows

### Basic Processing Pipeline

```bash
# 1. Run benchmarks
medarc-eval bench --config my-eval.yaml

# 2. Process results
medarc-eval process

# 3. Compute win rates
medarc-eval winrate
```

### CI/CD Pipeline

```bash
# Non-interactive processing with cleanup
medarc-eval process \
  --runs-dir ./benchmark-outputs \
  --output-dir ./processed \
  --clean \
  --yes \
  --max-workers 16
```

### Incremental Updates

```bash
# Process only new runs (default behavior)
medarc-eval process

# env_index.json tracks what's already processed
```

### Replace Existing Outputs

Rebuild existing outputs for specific models or datasets without using `--clean`:

```bash
# Rebuild every processed dataset for one model
medarc-eval process --replace-model gpt-4o

# Rebuild every model for one dataset
medarc-eval process --replace-env medqa

# Rebuild only the intersection
medarc-eval process --replace-model gpt-4o --replace-env medqa
```

When both flags are present, processing only rebuilds outputs that match both filters.

## Troubleshooting

### "No runs found"

Check that:
1. `--runs-dir` points to the correct location
2. Runs have completed (check `run_manifest.json` `jobs[*].status` and `summary.completed` / `summary.total`)
3. Use `--status pending` or `--status running` to include non-completed jobs

### Missing data in output

By default, only jobs with `completed` status are included. In addition, `--max-run-missing-pct` skips run directories missing more than 2.5% of their expected job outputs (model/env combinations), based on `run_manifest.json` `summary.completed` / `summary.total`. This is a manifest-level gate; it does not validate `results.jsonl` row counts.

Use `--max-run-missing-pct 100` to disable the gate, or pass explicit `--status` values to include other statuses.

### Integrity-check failures for existing parquet files

If processing stops with an error like:

```text
Existing processed output ... has N parquet rows but env_index.json records M.
```

the local processed snapshot is inconsistent. Fix it by rebuilding the affected output:

```bash
medarc-eval process --replace-model gpt-4o --replace-env medqa
```

Or rebuild everything:

```bash
medarc-eval process --clean --yes
```

## Next Steps

After processing, [compute win rates](medarc-eval-winrate.md) to compare model performance.
