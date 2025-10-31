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

Use the `benchmark` command to run a series of model/environment evaluations sequentially. Invoke it with `--jobs <path>`; the referenced YAML must define models inline (either in a top-level `models` list or directly on each job entry) and can reference environments inline or via a sibling `envs/` directory. By default, run artifacts are written to `<jobs_dir>/../runs`. A single-file example looks like this:

```yaml
name: medarc-baseline
output_dir: runs
envs:
  - id: medqa
    module: medqa
    num_examples: 25
    rollouts_per_example: 1
jobs:
  - model: gpt-4.1-mini
    temperature: 0.1
    env_args:
      judge_name: gpt-4.1-mini
    env: medqa
```

Inline models accept the same parameters that previously lived under `params` (`api_base_url`, `api_key_var`, `headers`, `sampling_args`, `env_overrides`, connection limits, etc.). If you prefer to reuse a model across several jobs, add it once under a top-level `models` list:

```yaml
models:
  - id: gpt-4.1-mini
    model: gpt-4.1-mini
    api_base_url: https://api.openai.com/v1
    timeout: 120

jobs:
  - model: gpt-4.1-mini
    envs: [medqa, medbullets]
```

A job without an explicit `name` defaults to the `model` string (or `model_id` when provided). The legacy `configs/models/*.yaml` files and `models_dir` option are no longer supported; migrate any remaining entries by copying their contents into either a top-level `models` list or directly into each job definition as shown above.

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

### Matrix expansion

Environment entries can fan out into multiple variants by adding a `matrix` mapping. Each key lists the values to vary, and the CLI takes the cartesian product to build derived configs. Values matching `EnvironmentConfig` fields (for example `num_examples`, `rollouts_per_example`, `max_concurrent`) are applied at the top level; all other keys land in `env_args`. Use `null` in a value list to keep the parent value for that combo.

```yaml
envs:
  - id: medconceptsqa-base
    module: medconceptsqa
    num_examples: -1
    env_args:
      shuffle_answers: true
    matrix:
      difficulty: [easy, medium, hard]
      shuffle_seed: [1618, 9331]
    matrix_id_format: "{base}-{difficulty}-s{shuffle_seed}"
```

The example above expands into six variants (`medconceptsqa-base-easy-s1618`, …, `medconceptsqa-base-hard-s9331`) that inherit the shared settings. Optional helpers:

- `matrix_exclude`: list of partial assignments to drop (e.g., `[{difficulty: easy, shuffle_seed: 9331}]`).
- `matrix_id_format`: custom template; `{base}` plus each matrix key is available. Without it, IDs default to `base-key-value`.

Entries without `matrix` continue to work unchanged. The loader validates duplicate IDs and unknown exclusions so mistakes surface early.

### Split configuration files

For larger suites you can still keep environment definitions in dedicated YAML files while keeping models inline. A common layout is:

```
configs/
├── envs/
│   ├── medqa.yaml
│   └── medcasereasoning.yaml
└── jobs.yaml
```

`envs/medqa.yaml`
```yaml
id: medqa
module: medqa
num_examples: 25
rollouts_per_example: 1
```

`jobs.yaml`
```yaml
name: medarc-suite
envs: ./envs
jobs:
  - model: gpt-4.1-mini
    api_base_url: https://api.openai.com/v1
    envs:
      - medqa
      - medcasereasoning
  - model: medarc/oss-judge
    model_id: oss-judge
    headers:
      X-Trace-Id: judge-run
    env: medqa
```

Run everything with:

```bash
uv run benchmark run --jobs configs/jobs.yaml
```

## Export evaluation runs to Parquet

The exporter CLI assembles completed run artifacts into environment-level Parquet datasets suitable for analytics or Hugging Face uploads.

```bash
uv run medarc-export \
  --runs-dir runs \
  --output-dir exports \
  --filter-status succeeded \
  --partition-by model \
  --dry-run
```

- `--filter-status` restricts discovery to manifest statuses of interest (defaults to all entries).
- `--dry-run` gathers schema details without writing files; combine with `--schema-only` to skip Parquet output entirely.
- `--partition-by` splits each environment dataset into Parquet files keyed by the provided columns (e.g., `model`, `job_run_id`).
- Prompt and completion payloads are excluded by default to keep files compact; pass `--include-io` to retain them.
- `--validate` enables sanity checks on row counts compared to metadata; add `--strict` to treat warnings as errors.
- `--overwrite` replaces existing environment export directories inside `--output-dir`.

When not in `--dry-run` or `--schema-only` mode, the CLI writes one directory per `env_id`, each containing either a single `data.parquet` file or partitioned files (e.g., `model-gpt-4_1-mini.parquet`). An `env_index.json` manifest summarises row counts, partition columns, and dataset paths for downstream tooling.

The loader expands each referenced file (environment mappings or the shared jobs list) before scheduling the run. Paths are resolved relative to the jobs file.

- When `envs` points to a directory, every `*.yaml` / `*.yml` file inside is loaded in sorted order. Paths are resolved relative to the jobs file first, then relative to the repository root.
- Inline models must be declared either at the job level or under a top-level `models` list within the same YAML document.
- When a job mapping supplies `envs`, the CLI fans out one evaluation per environment; omit `envs` (and `env`) to target every environment defined in the run.
- Optional `env_args`, `sampling_args`, and `seed` entries on the job apply to each generated evaluation. If an environment expects a `seed` argument, include it inside `env_args` explicitly.
- Override defaults with CLI flags such as `--envs`, `--env-dir-path`, and `--endpoints-path` when you want to load definitions from a different location.
- Job folders are named after the job's `name`; when omitted, the CLI generates `<model>-<random>-<env>` automatically.
