# MedARC Medical Language Model Environments

This repository is used to build verifiers environments and tools for the MedARC medical language model project.

It also contains the medarc-verifiers package, which provides additional tools for creating verifiers environments.

## Getting Started with Verifiers Environments

The steps below guide you through creating a new environment package under `environments/[my-new-env]`, installing it locally, testing it with Verifiers tooling, and optionally publishing it through Prime Intellect's Environments Hub.

### 1. Prerequisites
- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The [`prime` CLI](https://github.com/PrimeIntellect-ai/prime-cli) for scaffolding and publishing
- An OpenAI-compatible API key (export it as `OPENAI_API_KEY`) or OpenAI compatible model for testing the environment with `vf-eval`

### 2. Setup

Create and activate a virtual environment, then install the required tooling:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv tool install prime
uv pip install verifiers
```

After this setup the `prime env`, `vf-install`, and `vf-eval` commands will be available (or runnable via `uv run <command>`).

### 3. Create a New Environment
Always place new Verifiers packages inside `environments/my-new-env`. The Prime CLI ensures this by default:

```bash
# from the repository root
prime env init my-new-env
```

The template produces:
```
environments/my_new_env/
├── my_new_env.py
├── pyproject.toml
└── README.md
```

Edit `my_new_env.py` to configure datasets, parsers, and rubrics, and update the package metadata in `pyproject.toml` (name, version, dependencies, tags, etc.).

If the `prime env init` command doesn't add it, you'll want to add the following prime env metadata so prime/verifiers knows where the environment is in a flat repo:

```toml
[tool.prime.environment]
loader = "my_new_env:load_environment"
display_name = "My New Env"
visibility = "PUBLIC"
```

### 4. Install the Environment for Local Development
Install your new environment in editable mode so changes are picked up immediately:

```bash
vf-install my-new-env
# equivalent to:
# uv pip install -e ./environments/my_new_env
```

You can now import it from Python or let Verifiers discover it with `verifiers.load_environment("my-new-env")`.

### 5. Smoke-Test with `vf-eval`
Run a small batch of rollouts to confirm the environment behaves as expected. Set `OPENAI_API_KEY` (or whichever OpenAI client compatible credentials you plan to use) before invoking the CLI.

```bash
export OPENAI_API_KEY=sk-...
vf-eval my-new-env -m gpt-4.1-mini -n 5 -s
```

A few useful arguments:

- -m selects the inference model
- -n controls dataset size
- -s saves results locally.

Use vf-eval -h for the full set of options (rollouts per example, max concurrency, etc.)

During development you can iterate quickly by tweaking prompts, parser logic, or reward functions, reinstalling with `vf-install` if dependencies change, and rerunning `vf-eval` to view the results.

After running with `-s`, inspect saved runs with `vf-tui`, which provides a terminal UI for browsing prompts, completions, and rewards under the generated `outputs/evals` folders.

## Using an Existing MedARC Environment

Once your tooling is set up you can install MedARC-maintained environments directly from the Prime Environments Hub (for example [`medarc/medcasereasoning`](https://app.primeintellect.ai/dashboard/environments/medarc/medcasereasoning) or [`medarc/metamedqa`](https://app.primeintellect.ai/dashboard/environments/medarc/metamedqa)).

- **Install from the Hub.** Run `prime env install medarc/medcasereasoning` to pull the latest published version (add `@version` to pin a release).
- **Run an evaluation.** Execute `vf-eval medcasereasoning -m gpt-4.1-mini -n 10 -s` to generate a small batch of rollouts.
- **Load programmatically.** Environments installed via the Hub are importable like any other Verifiers module:

  ```python
  import verifiers as vf

  env = vf.load_environment("medcasereasoning", split="validation")
  results = env.evaluate(model_client, "gpt-4.1-mini", num_examples=5)
  ```

## medarc-eval CLI

`medarc-eval` wraps the upstream `vf-eval` flow, adding environment-specific flags and batch orchestration. See [full documentation](docs/medarc-eval.md).

| Command | Description |
|---------|-------------|
| [`medarc-eval <ENV>`](docs/medarc-eval-single-run.md) | Run a single benchmark with auto-discovered environment flags |
| [`medarc-eval bench`](docs/medarc-eval-bench.md) | Run multiple benchmarks from a YAML config with resume support |
| [`medarc-eval process`](docs/medarc-eval-process.md) | Convert raw outputs to parquet for analysis |
| [`medarc-eval winrate`](docs/medarc-eval-winrate.md) | Compute HELM-style win rates across models |

### Quick Start

```bash
# Run a single benchmark
uv run medarc-eval medqa -m gpt-4.1-mini -n 25

# Run batch evaluations from config
uv run medarc-eval bench --config configs/job-gpt-oss-20b.yaml

# Process results and compute win rates
uv run medarc-eval process
uv run medarc-eval winrate
```

### Environment-Specific Flags

Each environment's `load_environment()` parameters become CLI flags automatically:

```bash
# Discover available flags
uv run medarc-eval longhealth --help

# Use environment-specific options
uv run medarc-eval longhealth --task task1 --shuffle-answers -m gpt-4.1-mini -n 10
```

For complex arguments (dicts, nested structures), use `--env-args`:

```bash
uv run medarc-eval careqa --env-args '{"split": "open", "judge_model": "gpt-4o"}'
```

## Batch Evaluations

Use `medarc-eval bench` to run multiple model × environment evaluations from a config file. See [full batch mode documentation](docs/medarc-eval-bench.md).

```yaml
name: gpt-oss-20b-med

models:
  gpt-oss-20b:
    model: openai/gpt-oss-20b
    api_base_url: http://localhost:8000/v1
    sampling_args:
      temperature: 1.0
      reasoning_effort: medium

jobs:
  - model: gpt-oss-20b
    env: [m_arc, medcalc_bench, medxpertqa]
```

```bash
# Run the batch
uv run medarc-eval bench --config configs/job-gpt-oss-20b.yaml

# Preview without executing
uv run medarc-eval bench --config configs/job-gpt-oss-20b.yaml --dry-run
```

Batch mode supports automatic resume, job manifests, and matrix sweeps for parameter grids. See the [batch mode documentation](docs/medarc-eval-bench.md) for config file format, resume/restart options, and advanced features.

### Matrix Sweeps

Environment configs support matrix expansion for parameter grid runs:

```yaml
- id: medconceptsqa
  module: medconceptsqa
  num_examples: -1
  env_args:
    shuffle_answers: true
  matrix:
    difficulty: [easy, medium, hard]
    shuffle_seed: [1618, 9331]
  matrix_id_format: "{base}-{difficulty}-s{shuffle_seed}"
```

This expands into six variants (`medconceptsqa-base-easy-s1618`, …). See [batch mode docs](docs/medarc-eval-bench.md) for full details on matrix expansion, exclusions, and split config files.

## Processing and Win Rates

After running benchmarks, convert results to parquet and compute model comparisons:

```bash
# Process raw outputs to parquet
uv run medarc-eval process

# Compute HELM-style win rates
uv run medarc-eval winrate
```

See [processing documentation](docs/medarc-eval-process.md) and [win rate documentation](docs/medarc-eval-winrate.md) for configuration options, HuggingFace integration, and output formats.
