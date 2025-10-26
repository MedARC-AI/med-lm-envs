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

## medarc-eval CLI command

`medarc-eval` wraps the upstream `vf-eval` flow and adds environment-specific flags generated from each environment's `load_environment` signature to the CLI instead of requiring a json blob via `--env-args`.

### Quick start

```bash
uv run medarc-eval medqa -m gpt-4.1-mini -n 5
```

### Discover environment flags

```bash
uv run medarc-eval medbullets --help
```

### Mix explicit flags with JSON

```bash
uv run medarc-eval medbullets --num-options 4 --env-args '{"shuffle": true}'
```

Explicit flags always override JSON input. For list parameters, repeat the flag to replace the default entirely:

```bash
uv run medarc-eval longhealth --section cardio --section neuro
```

Use `--env-args` for complex structures (dicts, nested generics) that cannot be mapped to simple flags:

```bash
uv run medarc-eval medagentbench --env-args '{"config": {"mode": "fast"}}'
```

Print the detected environment schema:

```bash
uv run medarc-eval mmlu_pro_health --print-env-schema
```

## Benchmark CLI

Use the `benchmark` command to run a series of model/environment evaluations sequentially. Invoke it with `--jobs <path>`; the referenced YAML can inline models and environments or they can live alongside the file under `models/` and `envs/`. By default, run artifacts are written to `<jobs_dir>/../runs`. A single-file example looks like this:

```yaml
name: medarc-baseline
output_dir: runs
models:
  - id: gpt-4.1-mini
    params:
      model: gpt-4.1-mini
      sampling_args:
        temperature: 0.1
      env_args:
        judge_name: gpt-4.1-mini
envs:
  - id: medqa
    module: medqa
    num_examples: 25
    rollouts_per_example: 1
jobs:
  - model: gpt-4.1-mini
    env: medqa
    seed: 123
```

If `jobs` is omitted the CLI evaluates every model × environment pair defined in the config. Run a dry run to inspect the expanded matrix:

```bash
uv run benchmark dry-run --jobs configs/jobs.yaml
```

Execute the plan:

```bash
uv run benchmark run --jobs configs/jobs.yaml
```

By default the CLI installs each referenced environment before execution using the same logic as `vf-install`. This respects the resolved `env_dir_path`, ensuring local edits are available to the evaluator. Disable this behavior with `--no-install-envs` if you have already installed the packages or want to manage environments manually.

Each run reuses the `vf-eval` pipeline and writes artifacts into `<output_dir>/<run_id>/<job_id>/`. Job identifiers are deterministic (`<model>-<env>[-<name>]`) so reruns reuse the same directory layout. The CLI persists run metadata in `<output_dir>/<run_id>/run_manifest.json`, allowing interrupted runs to resume without recomputing completed jobs.

Resume an existing run (skipping finished jobs):

```bash
uv run benchmark run --jobs configs/jobs.yaml --resume medarc-baseline-20240101-120000
```

Force all jobs to re-run, even when artifacts already exist:

```bash
uv run benchmark run --jobs configs/jobs.yaml --resume medarc-baseline-20240101-120000 --force
```

The manifest merges updated configurations on resume, so adding a new model or environment entry will schedule only the new combinations while retaining prior results in `run_summary.json`.

### Split configuration files

For larger suites you can keep per-model, per-environment, and job matrices in separate YAML documents:

```
configs/
├── models/
│   ├── gpt4-mini.yaml
│   └── medarc-judge.yaml
├── envs/
│   ├── medqa.yaml
│   └── medcasereasoning.yaml
└── jobs.yaml
```

`gpt4-mini.yaml`
```yaml
id: gpt-4.1-mini
params:
  model: gpt-4.1-mini
```

`medqa.yaml`
```yaml
id: medqa
num_examples: 25
rollouts_per_example: 1
```

`jobs.yaml`
```yaml
- model: gpt-4.1-mini
  envs:
    - medqa
    - medcasereasoning
- model: medarc-judge
  env: medqa
  seed: 314
```
Run everything with:

```bash
uv run benchmark run --jobs configs/jobs.yaml
```

The loader expands each referenced file (individual model/env mappings or the shared jobs list) before scheduling the run. Paths are resolved relative to the jobs file.

- When `models` or `envs` point to directories, every `*.yaml` / `*.yml` file inside is loaded in sorted order. Paths are resolved relative to the jobs file first, then relative to the repository root.
- When a job mapping supplies `envs`, the CLI fans out one evaluation per environment; omit `envs` (and `env`) to target every environment defined in the run.
- Optional `env_args`, `sampling_args`, and `seed` entries on the job apply to each generated evaluation. If an environment expects a `seed` argument, include it inside `env_args` explicitly.
- Override defaults with CLI flags such as `--models`, `--envs`, `--env-dir-path`, and `--endpoints-path` when you want to load definitions from a different location.
- Job folders are named after the job's `name`; when omitted, the CLI generates `<model>-<random>-<env>` automatically.
