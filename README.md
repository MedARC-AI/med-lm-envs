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

## Token Tracking

Token usage is automatically tracked when using `medarc-eval`. Each result/rollout includes a `token_usage` column with nested dict:

```json
{
  "token_usage": {
    "model": {"prompt": 450, "completion": 280, "total": 730},
    "judge": {"prompt": 3200, "completion": 150, "total": 3350},
    "total": {"prompt": 3650, "completion": 430, "total": 4080}
  }
}
```

### Using with medarc-eval (Recommended)

Token tracking works automatically:

```bash
uv run medarc-eval medqa -m gpt-4.1-mini -n 5 -s
```

### Using with vf-eval

To enable token tracking with `vf-eval`, add `medarc_verifiers` as a dependency in your environment's `pyproject.toml`:

```toml
[project]
dependencies = [
    "verifiers>=0.1.2.post0",
    "medarc_verifiers>=0.1.0",
]

[tool.uv.sources]
medarc_verifiers = { git = "https://github.com/MedARC-AI/med-lm-envs" }
```

Then reinstall the environment:

```bash
uv pip install -e ./environments/your-env
vf-eval your-env -m gpt-4.1-mini -n 5 -s
```

### Disabling Token Tracking

```bash
export MEDARC_DISABLE_TOKEN_TRACKING=true
```

### Notes

- Works with any OpenAI-compatible provider
- Tokens extracted from API `response.usage` field
- If provider doesn't return usage data, defaults to 0
- Model tokens include all inference API calls
- Judge tokens include all LLM-as-judge calls via `judge()` method (e.g., FactScore: 6-20 verification calls per example)
- **Note**: Some judge implementations (e.g., FactScore claim extraction) make additional API calls (claim extraction) that are currently not tracked not part of judge() calls or get stored in state["responses"]. These represent a small overhead (~10-20% of total judge tokens) and are present in existing implementations like MedRedQA, keep in mind when calculating.