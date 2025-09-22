# MedARC Medical Language Model Evaluations

This repository is used to build evaluations for the MedARC medical language model project.

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