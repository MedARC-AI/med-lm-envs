# Batch Mode

Run multiple benchmarks across multiple models using a configuration file. Batch mode handles job scheduling, progress tracking, and automatic resume.

Each job invokes the verifiers [`vf-eval`](https://github.com/primeintellect-ai/verifiers) evaluation loop under the hood, with configuration-driven environment and sampling arguments.

## Quick Start

```bash
# Run all jobs from config
medarc-eval bench --config configs/job-gpt-oss-20b.yaml

# Preview what would run
medarc-eval bench --config configs/job-gpt-oss-20b.yaml --dry-run
```

## Writing a Config File

A minimal config defines models and which benchmarks to run:

```yaml
name: gpt-oss-20b-med

models:
  gpt-oss-20b:
    model: openai/gpt-oss-20b
    api_base_url: http://localhost:8000/v1
    sampling_args:
      temperature: 1.0
      top_p: 1.0
      top_k: 0
      reasoning_effort: medium

jobs:
  - model: gpt-oss-20b
    env:
      - m_arc
      - medcalc_bench
      - medxpertqa
```

This creates 3 jobs: gpt-oss-20b evaluated on m_arc, medcalc_bench, and medxpertqa.

### Config Structure

| Field | Description |
|-------|-------------|
| `name` | Human-readable run name |
| `output_dir` | Where to save results (default: `runs/raw`) |
| `models` | Map of model ID → model configuration |
| `jobs` | List of model + environment combinations to run |

### Model Configuration

```yaml
models:
  gpt-oss-20b:
    model: openai/gpt-oss-20b       # Model identifier
    api_base_url: http://localhost:8000/v1  # API endpoint (local or remote)
    api_key_var: OPENAI_API_KEY     # Optional: env var for API key
    max_concurrent: 10              # Optional: parallel request limit
    timeout: 120                    # Optional: request timeout (seconds)
    sampling_args:
      temperature: 1.0
      top_p: 1.0
      reasoning_effort: medium
```

### Environment Configuration

Benchmarks are configured in `configs/envs/`. Each file defines one or more environment variants:

```yaml
# configs/envs/medqa.yaml
- id: medqa
  module: medqa
  num_examples: -1              # -1 = all examples
  rollouts_per_example: 1
  env_args:
    shuffle_answers: true
    shuffle_seed: 1618
```

#### Matrix Sweeps

Run the same benchmark with different parameter combinations:

```yaml
- id: medqa
  module: medqa
  num_examples: -1
  env_args:
    shuffle_answers: true
  matrix:
    shuffle_seed: [1618, 9331, 2718]
  matrix_id_format: "{base}-seed{shuffle_seed}"
```

This creates three variants: `medqa-seed1618`, `medqa-seed9331`, `medqa-seed2718`.

## Resume and Restart

Batch mode automatically tracks job status and can resume interrupted runs.

### Automatic Resume (Default)

When you re-run the same config, completed jobs are skipped:

```bash
# First run - runs all jobs
medarc-eval bench --config my-config.yaml

# Interrupted, re-run - skips completed jobs
medarc-eval bench --config my-config.yaml
```

### Force Fresh Run

```bash
# Disable auto-resume, create new run directory
medarc-eval bench --config my-config.yaml --no-auto-resume

# Re-run everything, even completed jobs
medarc-eval bench --config my-config.yaml --force
```

### Restart from Previous Run

Copy completed jobs from an old run to seed a new one:

```bash
medarc-eval bench --config updated-config.yaml --restart old-run-id
```

### Re-run Specific Environments

```bash
# Re-run only medqa jobs (keep other completed jobs)
medarc-eval bench --config my-config.yaml --forced medqa

# Re-run multiple environments
medarc-eval bench --config my-config.yaml --forced medqa,pubmedqa
```

## Common Flags

### Job Selection

| Flag | Description |
|------|-------------|
| `--config PATH` | **Required.** Path to config YAML |
| `--job-id ID` | Run only specific job(s) by ID (repeatable) |
| `--dry-run` | Show plan without executing |

### Output Control

| Flag | Description |
|------|-------------|
| `--output-dir PATH` | Override output directory |
| `--run-id ID` | Force specific run directory name |
| `--name NAME` | Override run name in manifest |

### Override All Jobs

| Flag | Description |
|------|-------------|
| `--env-args JSON` | Override environment args for all jobs |
| `--sampling-args JSON` | Override sampling args for all jobs |
| `--max-concurrent N` | Override concurrency for all jobs |
| `--timeout SEC` | Override timeout for all jobs |

## Output Structure

```
runs/raw/<run_id>/
├── run_manifest.json           # Run metadata, job status, checksums
├── <job_id>/
│   ├── results.jsonl           # Per-example results
│   ├── summary.json            # Aggregate metrics
│   └── metadata.json           # Job configuration snapshot
└── <job_id>/
    └── ...
```

The manifest tracks:
- Job status (pending, running, completed, failed)
- Configuration checksums for resume detection
- Timing information
- Output paths

## Example Workflows

### Evaluate Multiple Models on Core Benchmarks

```yaml
name: model-comparison

models:
  gpt-oss-20b:
    model: openai/gpt-oss-20b
    api_base_url: http://192.168.1.152:8000/v1
    sampling_args:
      temperature: 1.0
      reasoning_effort: medium

  gpt-oss-20b-low:
    model: openai/gpt-oss-20b
    api_base_url: http://192.168.1.152:8000/v1
    sampling_args:
      temperature: 0.7
      reasoning_effort: low

jobs:
  - model: gpt-oss-20b
    env: [m_arc, medcalc_bench, medxpertqa]
  - model: gpt-oss-20b-low
    env: [m_arc, medcalc_bench, medxpertqa]
```

### Override Parameters at Runtime

```bash
# Lower concurrency for rate-limited API
medarc-eval bench --config my-config.yaml --max-concurrent 5

# Change temperature for all jobs
medarc-eval bench --config my-config.yaml --sampling-args '{"temperature": 0.5}'
```

## Next Steps

After batch runs complete:
1. [Process results](medarc-eval-process.md) into parquet format
2. [Compute win rates](medarc-eval-winrate.md) to compare models
