# Getting Started

This guide covers the developer workflow for Medmarks benchmark environments and the `medarc-verifiers` tooling in this repository.

## Getting Started with Verifiers Environments

The steps below guide you through creating a new environment package under `environments/[my-new-env]`, installing it locally, testing it with Verifiers tooling, and optionally publishing it through Prime Intellect's Environments Hub.

### 1. Prerequisites

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The [`prime` CLI](https://github.com/PrimeIntellect-ai/prime-cli) for scaffolding and publishing
- An OpenAI-compatible API key, exported as `OPENAI_API_KEY`, or another OpenAI-compatible model endpoint for testing environments with `vf-eval`

### 2. Setup

Create and activate a virtual environment, then install the required tooling:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv tool install prime
```

After this setup the `prime env`, `vf-install`, `vf-eval`, and `medarc-eval` commands will be available, or runnable via `uv run <command>`.

### 3. Create a New Environment

Always place new Verifiers packages inside `environments/my-new-env`. The Prime CLI ensures this by default:

```bash
# from the repository root
prime env init my-new-env
```

The template produces:

```text
environments/my_new_env/
|-- my_new_env.py
|-- pyproject.toml
`-- README.md
```

Edit `my_new_env.py` to configure datasets, parsers, and rubrics, and update the package metadata in `pyproject.toml` with the package name, version, dependencies, tags, and related fields.

If the `prime env init` command does not add it, add the following Prime environment metadata so Prime and Verifiers know where the environment lives in a flat repo:

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

Run a small batch of rollouts to confirm the environment behaves as expected. Set `OPENAI_API_KEY`, or whichever OpenAI-compatible credentials you plan to use, before invoking the CLI.

```bash
export OPENAI_API_KEY=sk-...
vf-eval my-new-env -m gpt-4.1-mini -n 5 -s
```

A few useful arguments:

- `-m` selects the inference model.
- `-n` controls dataset size.
- `-s` saves results locally.

Use `vf-eval -h` for the full set of options, including rollouts per example and max concurrency.

During development you can iterate quickly by tweaking prompts, parser logic, or reward functions, reinstalling with `vf-install` if dependencies change, and rerunning `vf-eval` to view the results.

After running with `-s`, inspect saved runs with `vf-tui`, which provides a terminal UI for browsing prompts, completions, and rewards under the generated `outputs/evals` folders.

## Using an Existing Medmarks Environment

Once your tooling is set up you can install MedARC-maintained environments directly from the Prime Environments Hub, for example [`medarc/medcasereasoning`](https://app.primeintellect.ai/dashboard/environments/medarc/medcasereasoning) or [`medarc/metamedqa`](https://app.primeintellect.ai/dashboard/environments/medarc/metamedqa).

- Install from the Hub: run `prime env install medarc/medcasereasoning` to pull the latest published version. Add `@version` to pin a release.
- Run an evaluation: execute `vf-eval medcasereasoning -m gpt-4.1-mini -n 10 -s` to generate a small batch of rollouts.
- Load programmatically:

```python
import verifiers as vf

env = vf.load_environment("medcasereasoning", split="validation")
results = env.evaluate(model_client, "gpt-4.1-mini", num_examples=5)
```

## medarc-eval CLI

`medarc-eval` wraps the upstream `verifiers` eval flow, adding environment-specific flags and a TOML bench workflow. See the [full documentation](medarc-eval.md).

| Command | Description |
|---------|-------------|
| [`medarc-eval <ENV>`](medarc-eval-single-run.md) | Run a single benchmark with auto-discovered environment flags |
| [`medarc-eval bench`](medarc-eval-bench.md) | Run upstream TOML eval configs with deterministic MedARC paths |
| [`medarc-eval process`](medarc-eval-process.md) | Convert eval outputs to parquet for analysis |
| [`medarc-eval winrate`](medarc-eval-winrate.md) | Compute HELM-style win rates across models |

### Quick Start

```bash
# Run a single benchmark
uv run medarc-eval medqa -m gpt-4.1-mini -n 25

# Run batch evaluations from config
uv run medarc-eval bench --config configs/eval/medmarks-smoke.toml

# Process results and compute win rates
uv run medarc-eval process --runs-dir runs/evals
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

For complex arguments such as dicts and nested structures, use `--env-args`:

```bash
uv run medarc-eval careqa --env-args '{"split": "open", "judge_model": "gpt-4o"}'
```

## Batch Evaluations

Use `medarc-eval bench` to run upstream `verifiers` TOML eval configs sequentially with deterministic MedARC output paths. See the [bench mode documentation](medarc-eval-bench.md).

```toml
model = "openai/gpt-4.1-mini"
save_results = true
output_dir = "runs/evals"

[[eval]]
env_id = "medqa"
num_examples = 25
rollouts_per_example = 1
env_args = { shuffle_answers = true, shuffle_seed = 1618 }
```

```bash
# Run the batch
uv run medarc-eval bench --config configs/eval/medmarks-verified.toml

# Preview without executing
uv run medarc-eval bench --config configs/eval/medmarks-verified.toml --dry-run
```

Bench mode resumes matching deterministic result directories and supports `[[ablation]]` sweeps for parameter grids. The removed YAML job/manifest runner is documented only in the migration notes in the [bench mode docs](medarc-eval-bench.md).

### Ablation Sweeps

Use upstream TOML ablations for parameter grid runs:

```toml
[[ablation]]
env_id = "medconceptsqa"
num_examples = -1
env_args = { shuffle_answers = true }

[ablation.sweep.env_args]
difficulty = ["easy", "medium", "hard"]
shuffle_seed = [1618, 9331]
```

This expands into deterministic variant directories under `runs/evals/<model>/medconceptsqa/`. See the [bench mode docs](medarc-eval-bench.md) for details.

## Processing and Win Rates

After running benchmarks, convert results to parquet and compute model comparisons:

```bash
# Process eval outputs to parquet
uv run medarc-eval process --runs-dir runs/evals

# Compute HELM-style win rates
uv run medarc-eval winrate
```

See the [processing documentation](medarc-eval-process.md) and [win rate documentation](medarc-eval-winrate.md) for configuration options, Hugging Face integration, and output formats.
