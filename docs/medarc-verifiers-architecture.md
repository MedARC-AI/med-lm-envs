# medarc_verifiers: Architecture & Workflow Overview

This is a coding agents guide to `medarc_verifiers/`.

## What `medarc_verifiers` is

`medarc_verifiers` wraps and extends the upstream `verifiers` evaluation
framework with:

- A unified CLI (`medarc-eval`) for medical benchmark environments.
- A TOML bench wrapper for sequential local benchmark runs with deterministic output paths.
- A processing pipeline that converts eval output artifacts into analysis-ready Parquet datasets.
- HELM-style win rate computation across models from processed outputs.
- Shared environment utilities for parsers, rewards, shuffling, and judging.

The current workflow is:

1. **Run** evals with single-run mode or TOML bench -> `runs/evals/<model>/<env>/...`
2. **Process** eval outputs -> `runs/processed/<model>/<env>.parquet` plus `env_index.json`
3. **Winrate** on processed outputs -> `runs/processed/winrate/*.json` and `*.csv`

Historical YAML-runner outputs under `runs/raw/<run_id>/...` must be converted
with `scripts/convert_legacy_raw_runs.py` before `medarc-eval process` can read
them. The YAML benchmark runner itself has been removed.

## Import Side Effects

Importing `medarc_verifiers` installs monkey patches into `verifiers` by default
(`medarc_verifiers/__init__.py`):

- **Judge cache namespacing**: cached judge responses are keyed by
  `base_url::model` so multi-judge runs do not collide
  (`medarc_verifiers/judging/judge_cache_fix.py`).

`token_usage` is produced by upstream `verifiers` output serialization and is
flattened into explicit columns during `medarc-eval process`.

## `medarc-eval` CLI

Entry point and router: `medarc_verifiers/cli/main.py`.

It supports:

- **Single-run mode**: `medarc-eval <ENV> ...`
  - The environment name must be the first token.
  - Implemented in `medarc_verifiers/cli/_single_run.py`.
- **TOML bench mode**: `medarc-eval bench --config <config.toml>`
  - Loads upstream `verifiers` TOML eval configs, expands ablations, plans
    deterministic output directories from selected raw configs, then runs evals
    sequentially through upstream execution.
  - Missing selected local environment packages are auto-installed by default
    from `--env-dir` (default `environments`) in isolated system temporary
    venvs with a `medarc-bench-venv-` prefix. Importable envs stay on the
    in-process path. `--no-auto-install` requires selected envs to already be
    importable.
  - Main implementation: `medarc_verifiers/cli/main.py`
  - Isolated auto-install helper: `medarc_verifiers/cli/isolated_env.py`
  - Isolated child runner: `medarc_verifiers/cli/bench_child.py`
  - Upstream eval boundary: `medarc_verifiers/cli/upstream_eval.py`
  - Deterministic identity/path helpers: `medarc_verifiers/cli/eval_identity.py`
- **Processing**: `medarc-eval process ...`
  - Pipeline wiring: `medarc_verifiers/cli/process/pipeline.py`
- **Win rates**: `medarc-eval winrate ...`
  - Runner: `medarc_verifiers/cli/winrate/runner.py`
  - Core math: `medarc_verifiers/cli/winrate/api.py`

Shared CLI constants live in `medarc_verifiers/cli/_constants.py`.

## Dynamic Env Flags

Single-run mode introspects each environment's `load_environment()` signature
and docstring to generate argparse flags dynamically:

- Introspection and validation: `medarc_verifiers/cli/utils/env_args.py`

That is why `medarc-eval longhealth --help` shows environment-specific flags
even though they are not hardcoded. For anything too complex for flags,
single-run and TOML bench both support:

- `--env-args '{...json...}'`
- `--env-arg key=value` (repeatable; smart type coercion)

Override parsing lives in `medarc_verifiers/cli/utils/overrides.py`.

## TOML Bench Config Semantics

Bench configs use upstream `verifiers` TOML shape: top-level defaults plus one
or more `[[eval]]` entries. Upstream `[[ablation]]` tables expand into repeated
eval configs. MedARC adds deterministic paths around selected raw eval configs
before importing env packages. Duplicate `(model, env)` outputs must use
explicit `variant_id` or `name` identity; the reserved default variant id is
`base`.

`env_args` precedence is low to high:

1. Environment package `[tool.verifiers.eval]` defaults, when discoverable
2. TOML top-level defaults
3. Per-`[[eval]]` values
4. Expanded `[[ablation]]` values
5. CLI overrides (`--env-args` / `--env-arg`)

Environment package `[tool.verifiers.eval]` defaults are execution-time
defaults. They do not affect deterministic path planning or dry-run display,
because bench plans from TOML and CLI values before importing env packages.

`sampling_args` follow the same TOML -> eval -> ablation -> CLI override model,
then are sanitized once for the resolved Verifiers client type:

- Unknown parameters move under `extra_body` for compatible servers such as vLLM.
- OpenAI Chat Completions keeps `reasoning_effort` as a top-level request field.
- OpenAI Responses maps `reasoning_effort` to `reasoning = {"effort": ...}`.
- Anthropic Messages uses adaptive thinking only:
  `thinking = {"type": "adaptive"}` plus
  `output_config = {"effort": ...}`. Manual `budget_tokens` thinking configs
  are rejected before execution.
- Sanitizer: `medarc_verifiers/utils/sampling_args.py`
- Import boundary: `medarc_verifiers/cli/upstream_eval.py`
- Temporary merge/adaptation adapter behind that boundary:
  `medarc_verifiers/cli/verifiers_adapter.py`

The old YAML `models`, `envs`, `jobs`, matrix expansion, job builder, and
manifest planner modules have been deleted.

## Endpoints and Prime Inference

There are two related concepts:

1. **Endpoint registry**: optional aliases for endpoint URL and key env var.
   - Loader and cache: `medarc_verifiers/cli/utils/endpoint_utils.py`
   - CLI default path: `configs/endpoints.toml`
2. **Prime Inference overrides**:
   - Adds `X-Prime-Team-ID` from `PRIME_TEAM_ID`.
   - Selects `PRIME_API_KEY` when available for Prime Inference endpoints.
   - Enables usage reporting unless disabled by `MEDARC_INCLUDE_USAGE=false`.
   - Implementation: `medarc_verifiers/utils/prime_inference.py`

Relevant env vars:

- `OPENAI_API_KEY`
- `PRIME_API_KEY`, `PRIME_TEAM_ID`
- `MEDARC_INCLUDE_USAGE`

## Resume and Deterministic Paths

TOML bench writes eval outputs under deterministic directories:

- Non-variant evals: `runs/evals/<model>/<env>/base/`
- Variant evals: `runs/evals/<model>/<env>/<variant_id>/`

If neither `--output-dir` nor TOML `output_dir` is set, the output root
defaults to `runs/evals`. Existing valid outputs resume automatically: bench
passes the deterministic target as upstream `EvalConfig.resume_path` and trusts
upstream resume validation. Partial or malformed existing targets fail unless
`--force` archives the existing target and reruns.

For missing local envs, auto-install creates a temporary venv, mirrors the
current `medarc-verifiers` install into that venv, installs the target env
package, and only then prepares or archives the deterministic output directory.
Editable MedARC installs mirror the same checkout from package metadata.
Non-editable installs use `medarc-verifiers==<current-version>` and require
that distribution to be resolvable.

`medarc-eval bench` does not monkey-patch upstream metadata saving and does not
write MedARC identity into upstream `metadata.json`. Variant identity is the
deterministic path segment, so `variant_id` / `name` values must already be
path-safe.

Historical raw-run manifest schemas are not part of the runtime package. Use
`scripts/convert_legacy_raw_runs.py` as a one-off migration helper for old
`runs/raw` artifacts.

## Orchestrated vLLM Runs

Docs: `docs/medarc-orchestrate.md`.

`medarc-orchestrate` accepts the same upstream eval TOML job configs that
`medarc-eval bench` accepts. Runtime infrastructure is resolved from endpoint registry entries:

- `endpoints.toml` or `medmarks-endpoints.toml` stores aliases under `[[endpoint]]`; entries with
  `[endpoint.orchestrate]` are orchestratable.
- `eval_images.toml` stores eval-scoped auxiliary images selected by eval/env id.
- The same endpoint registry is used for exact `endpoint_id` matching and passed through to the worker bench command.

Task bundles live under `outputs/orchestrate/<run_id>/tasks/<task-slug>/` and
contain bundled `eval-config.toml`, internal `task.yaml`, registry snapshot TOML
files, allocation state, runtime logs, and a task-local `bench/` output root.
Workers always run `medarc-eval bench --config <task>/eval-config.toml --provider local --output-dir <task>/bench`; removed YAML-runner flags such as `--run-id`, `--restart`, and `--on-complete` are not used. Processing can scan these nested task-local bench outputs recursively from the orchestrator run root.

## Eval Outputs

TOML bench outputs include:

- `results.jsonl`: per-example rollouts
- `metadata.json`: eval configuration and metrics snapshot

The runner executes via `verifiers.utils.eval_utils.run_evaluation()` from
single-run mode and the TOML bench code in `medarc_verifiers/cli/main.py`.

## Processing Pipeline

Docs: `docs/medarc-eval-process.md`.

Entry point: `medarc_verifiers/cli/process/pipeline.py`.

Processing:

1. Discovers eval outputs from `runs/evals` by scanning directories containing
   `metadata.json` and `results.jsonl`.
2. Normalizes identity from upstream `metadata.json` and deterministic paths.
3. Loads rows from `results.jsonl`, drops large prompt/completion fields, and
   flattens `token_usage`.
4. Aggregates rows per model and environment, preserving variant ids.
5. Writes Parquet files plus `env_index.json` and `dataset_infos.json`.

Important modules:

- Discovery: `medarc_verifiers/cli/process/discovery.py`
- Metadata normalization: `medarc_verifiers/cli/process/metadata.py`
- Row loading: `medarc_verifiers/cli/process/rows.py`
- Aggregation: `medarc_verifiers/cli/process/aggregate.py`
- Writing/indexing: `medarc_verifiers/cli/process/writer.py`,
  `medarc_verifiers/cli/process/env_index.py`

## Win Rates

Docs: `docs/medarc-eval-winrate.md`.

`medarc-eval winrate` reads dataset inventory from `env_index.json`, averages
rollouts per `(example_id, model_id)`, and computes pairwise model comparisons.

- Dataset discovery: `medarc_verifiers/cli/winrate/runner.py`
- Core math and weighting policies: `medarc_verifiers/cli/winrate/api.py`
- Outputs: timestamped `winrates-<timestamp>.json` / `.csv` plus
  `latest.json` / `latest.csv`

## Environment Utilities

Frequently imported utilities under `environments/*`:

- Prompts and answer format constants: `medarc_verifiers/prompts.py`
- XML parser: `medarc_verifiers/parsers/xml_parser.py`
- JSON parser: `medarc_verifiers/parsers/json_parser.py`
- MCQ grading: `medarc_verifiers/rewards/multiple_choice_accuracy.py`
- HELM reward normalization: `medarc_verifiers/rewards/normalize_helm_reward.py`
- Deterministic MCQ shuffling: `medarc_verifiers/utils/randomize_multiple_choice.py`
- Judge helpers: `medarc_verifiers/utils/judge_helpers.py`

## Judging and Multi-Judge Support

Some environments use LLM-as-judge scoring. `medarc_verifiers` provides:

- Judge call wrapper: `medarc_verifiers/judging/judge_core.py`
- Multi-judge runner: `medarc_verifiers/judging/multi_judge.py`
- Verifiers-compatible rubric wrapper: `medarc_verifiers/judging/multi_judge_rubric.py`

## vLLM Orchestrator

Docs: `docs/medarc-orchestrate.md`.

`medarc-orchestrate` runs TOML bench configs against locally hosted vLLM
containers with GPU/port scheduling across Docker or Slurm+Pyxis runtimes.

- CLI entry: `medarc_verifiers/orchestrate/cli.py`
- Runtime loop: `medarc_verifiers/orchestrate/run.py`

It:

1. Launches vLLM containers.
2. Waits for readiness.
3. Runs `uv run medarc-eval bench --config <job.toml> --api-base-url <allocated> --provider local`.
4. Tracks orchestration state under `outputs/orchestrate/<run_id>/`.

## Where To Change Things

- CLI flags or routing:
  - `medarc_verifiers/cli/main.py`, `medarc_verifiers/cli/_single_run.py`
- TOML bench behavior, deterministic paths, or bench sidecar identity:
  - `medarc_verifiers/cli/main.py`, `medarc_verifiers/cli/eval_identity.py`,
    `medarc_verifiers/cli/upstream_eval.py`, `medarc_verifiers/cli/verifiers_adapter.py`
- Processed dataset schema:
  - `medarc_verifiers/cli/process/rows.py`, `medarc_verifiers/cli/process/writer.py`
- Winrate math/output:
  - `medarc_verifiers/cli/winrate/api.py`, `medarc_verifiers/cli/winrate/runner.py`
- Judging/provider behavior:
  - `medarc_verifiers/utils/judge_helpers.py`, `medarc_verifiers/utils/prime_inference.py`
