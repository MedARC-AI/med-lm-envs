# Processing Results

Convert raw benchmark outputs into analysis-ready parquet files. This step prepares data for win rate computation and other analyses.

## Quick Start

```bash
# Process outputs from the current TOML bench runner
medarc-eval process --runs-dir runs/evals --output-dir runs/processed

# Process legacy manifest outputs (default runs dir)
medarc-eval process

# Preview what would be processed
medarc-eval process --dry-run
```

## What Processing Does

1. **Discovers** eval outputs in `runs/evals/` by scanning output directories,
   and legacy manifest jobs in `runs/raw/`
2. **Extracts** results from each eval output directory
3. **Normalizes** data into a fixed output schema
4. **Writes** parquet files organized by model and environment
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

On-disk model and env path components are slugified, so filenames may not exactly match raw ids.

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--runs-dir PATH` | Directory containing raw run outputs. Use `runs/evals` for new TOML bench runs. | `runs/raw` |
| `--output-dir PATH` | Where to write processed files | `runs/processed` |
| `--max-workers N` | Parallel worker processes | 4 |
| `--dry-run` | Show what would be processed | - |
| `--yes` | Skip confirmation prompts | - |
| `--exclude-dataset NAME` | Skip processing specific datasets/env ids (repeatable) | - |
| `--exclude-model MODEL` | Skip processing specific model ids (repeatable) | - |

## Filtering Runs

### By Completion Status

For current TOML bench outputs, processing scans for directories containing
`metadata.json` and `results.jsonl`. When a model-level
`.medarc_eval_metadata.json` helper exists, processing uses it only to enrich
matching scanned paths with MedARC `env_id` / `variant_id`. Stale helper entries
are ignored. Ad hoc upstream outputs fall back to metadata/path inference.

For legacy YAML-runner outputs, `medarc-eval process` reads
`runs/raw/<run_id>/run_manifest.json` and only selects jobs whose manifest
status is `completed` by default.

Note: successful legacy jobs are written to `run_manifest.json` with
`status: completed`.

To override that default, pass one or more explicit status filters:

```bash
medarc-eval process --status completed --status failed
```

You can also gate partially complete outputs by missing `results.jsonl` rows:

```bash
# Default tolerance is 2.5 percent missing
medarc-eval process --max-results-missing-pct 2.5

# Effectively disable the gate
medarc-eval process --max-results-missing-pct 100
```

For TOML bench outputs, this gate uses `metadata.json` values for expected rows
and the observed `results.jsonl` row count. Model and environment identity come
from upstream metadata and path inference, with variant identity enriched by the
model-level helper when present:

- `expected_rows = num_examples * rollouts_per_example`
- `observed_rows = results.jsonl row count`

For legacy manifest outputs, the same gate uses manifest job fields:

- `expected_rows = num_examples * rollouts_per_example`
- `observed_rows = row_count`

It is computed per selected job record and enforced only on the latest selected run for each processed model/environment output. For legacy manifests, it does not use manifest `summary.completed` / `summary.total`, and it does not fall back to older runs if the latest one is too incomplete.

Selected records with missing `results.jsonl` fail processing immediately.

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
runs_dir: runs/evals

process:
  dir: processed
  max_workers: 8
  max_results_missing_pct: 2.5
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

- Top-level `runs_dir`: raw run root. Use `runs/evals` for new TOML bench outputs and `runs/raw` for legacy manifest outputs.
- Top-level `process:`: process-specific defaults.
- Optional top-level `winrate:`: embedded post-process winrate step.
- Optional top-level `hf:`: shared HF settings. For embedded winrate uploads, use `hf.winrate_dir`.

Path shortcuts:

- `process.dir` is shorthand for `process.output_dir`, resolved relative to the parent of `runs_dir`.
- `winrate.dir` is shorthand for the embedded winrate output directory, resolved under the processed output dir.

Example:

```yaml
runs_dir: runs/evals

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
runs_dir: runs/evals
process:
  dir: processed

hf:
  repo: your-org/medical-benchmarks
  branch: main
  token: ${HF_TOKEN}
  private: true
```

`hf.token` accepts either a literal token string or an environment reference like `$HF_TOKEN` / `${HF_TOKEN}`.

### Pull Before Processing

```bash
# Prompt before pulling
medarc-eval process --hf-repo your-org/data --hf-pull-policy prompt

# Always pull existing data first
medarc-eval process --hf-repo your-org/data --hf-pull-policy pull

# Start fresh (ignore remote)
medarc-eval process --hf-repo your-org/data --hf-pull-policy clean

# Resume a previously failed HF upload without pulling or cleaning
medarc-eval process --hf-repo your-org/data --hf-pull-policy continue-upload
```

`prompt` only prompts when the local processed dir is already non-empty. If the output dir is empty, process pulls the HF baseline immediately.

When `prompt` is used with a non-empty local processed dir, the menu may show:

- `pull`: download missing baseline data without deleting local files
- `clean`: redownload everything after deleting local files
- `upload`: keep local processed outputs and resume/upload pending HF artifacts

`upload` is shown only when local parquet files appear to be missing remotely or have a different remote `lfs.sha256`. Recovery uploads the union of:

- parquet files that were already pending before the current run started
- files touched by the current process run, including `env_index.json` and `dataset_infos.json` when rewritten

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
medarc-eval bench --config configs/eval/medmarks-verified.toml

# 2. Process results
medarc-eval process --runs-dir runs/evals

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
# Process only new TOML bench outputs
medarc-eval process --runs-dir runs/evals

# env_index.json tracks what's already processed
```

Incremental skipping only reuses an existing parquet when its footer metadata `source_runs` still matches the newly selected run ids and the existing row count still matches `env_index.json`.

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
2. For TOML bench outputs, each eval directory contains `results.jsonl` and `metadata.json`
3. For legacy manifest outputs, runs have completed (check `run_manifest.json` `jobs[*].status`)
4. Use `--status pending` or `--status running` to include non-completed legacy jobs

### Missing data in output

By default, current TOML bench outputs are selected from valid eval directories, while legacy manifest outputs include only jobs with `completed` status. In addition, `--max-results-missing-pct` fails if a selected latest job record is missing more than 2.5% of its expected `results.jsonl` rows. TOML bench outputs use eval metadata plus the observed JSONL row count; legacy manifest outputs use manifest job fields:

- `row_count`
- `num_examples`
- `rollouts_per_example`

The gate is per selected record, not per whole run manifest. If the latest selected run for a model/dataset is too incomplete, processing fails fast instead of silently falling back to an older run. Records with unknown expected rows or unknown `row_count` are not gated.

Use `--max-results-missing-pct 100` to disable the gate, or pass explicit `--status` values to include other statuses.

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
