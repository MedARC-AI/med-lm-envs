# Repository Guidelines

## Project Structure & Module Organization

- `medarc_verifiers/`: Core Python package (CLI entrypoints, parsers, rewards, orchestration utilities).
- `environments/<env>/`: Individual Verifiers environments (each is a small Python package with `<env>.py` and its own `pyproject.toml`).
- `configs/`: YAML configs for `medarc-eval bench` (job matrices, env configs, judge configs).
- `docs/`: Usage docs for `medarc-eval` and related workflows.
- `tests/`: `pytest` suite.

## Architecture Overview (Start Here)

- Detailed overview: `docs/medarc-verifiers-agents-overview.md`
- Quick workflow: eval → process → winrate
  - raw outputs: `runs/raw/<run_id>/...`
  - processed parquet: `runs/processed/<model>/<env>.parquet` + `runs/processed/env_index.json`
  - winrate outputs: `runs/winrate/latest.json` and `runs/winrate/latest.csv`
- CLI entrypoint/router: `medarc_verifiers/cli/main.py`
- Batch resume/restart state lives in `runs/raw/<run_id>/run_manifest.json`
- Environment `load_environment()` params become CLI flags (see `medarc-eval <env> --help`).
- Environment authoring utilities (used by `environments/*`):
  - parsing/prompts: `medarc_verifiers/parsers/`, `medarc_verifiers/prompts.py` (XML preferred; BOXED supported)
  - MCQ grading: `medarc_verifiers/rewards/multiple_choice_accuracy.py`
  - deterministic shuffling: `medarc_verifiers/utils/randomize_multiple_choice.py` (use `shuffle_seed`)
  - judging: `medarc_verifiers/judging/`, `medarc_verifiers/utils/judge_helpers.py`

## Build, Test, and Development Commands

- `uv venv --python 3.12 && source .venv/bin/activate`: Create/activate a local venv.
- `uv pip install -e .`: Install `medarc-verifiers` in editable mode.
- `vf-install <env>`: Install an environment from `environments/<env>/` in editable mode.
- `uv run medarc-eval <ENV> -m <MODEL> -n 5`: Run a small evaluation.
- `uv run medarc-eval bench --config configs/job.yaml`: Run a batch evaluation from a YAML config.
- `uv run pytest tests/`: Run the full test suite.
- `uv run ruff check medarc_verifiers/ && uv run ruff format medarc_verifiers/`: Lint/format.

## Adding or Updating Environments

- Create a new environment package: `prime env init my-new-env` (creates `environments/my_new_env/`).
- Ensure the environment `pyproject.toml` has `[tool.prime.environment]` with a correct `loader` (e.g., `my_new_env:load_environment`).
- Smoke test: `vf-install my-new-env && medarc-eval my-new-env -m gpt-4.1-mini -n 5`.

## Coding Style & Naming Conventions

- Python `>=3.11`; prefer type hints for new code.
- Formatting/linting: Ruff (configured in `pyproject.toml`, line length `120`).
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `test_*.py` for tests.

## Testing Guidelines

- Frameworks: `pytest` + `pytest-asyncio` for async code.
- Example: `uv run pytest tests/test_cli/test_process_winrate.py -k winrate`.

## Commit & Pull Request Guidelines

- Commit messages: Use a single short sentence (imperative mood, no period). Example: "Remove duplicate cli_env_args module"
- PRs: Short, scoped subjects with clear descriptions.

## Security & Configuration Tips

- Never commit secrets; use env vars like `OPENAI_API_KEY`, `PRIME_API_KEY`, and `PRIME_TEAM_ID`.
- Prime Inference usage reporting: `MEDARC_INCLUDE_USAGE=true/false` or `medarc-eval ... --include-usage/--no-include-usage`.
- Keep local evaluation artifacts out of git (e.g., `outputs/`).
