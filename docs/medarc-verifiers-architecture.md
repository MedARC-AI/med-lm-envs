# medarc_verifiers: Architecture & Workflow Overview

This is a coding agents guide to `medarc_verifiers/`.

## What `medarc_verifiers` is

`medarc_verifiers` is the repository’s Python package that wraps and extends the upstream `verifiers` evaluation framework with:

- A unified CLI (`medarc-eval`) for running many medical benchmark environments consistently.
- Batch orchestration with durable run manifests (resume/restart/force).
- A processing pipeline that converts raw run artifacts into analysis-ready Parquet datasets.
- HELM-style win rate computation across models from processed outputs.
- Shared building blocks used by environments (parsers, rewards, shuffling utilities, judge helpers).

At a high level, everything funnels into a three-stage workflow:

1. **Run** evals (single or batch) → `runs/raw/<run_id>/...`
2. **Process** raw outputs → `runs/processed/<model>/<env>.parquet` + `env_index.json`
3. **Winrate** on processed outputs → `runs/winrate/*.json` and `*.csv`

## Important side effects (auto-installed patches)

Importing `medarc_verifiers` installs monkey patches into `verifiers` by default (`medarc_verifiers/__init__.py`):

- **Judge cache namespacing**: cached judge responses are keyed by `base_url::model` so multi-judge runs don’t collide (`medarc_verifiers/judging/judge_cache_fix.py`).

`token_usage` is now produced by upstream `verifiers` output serialization and is flattened into explicit columns during `medarc-eval process`.

## `medarc-eval` CLI: modes and code layout

Entry point and router: `medarc_verifiers/cli/main.py`.

It supports:

- **Single-run mode**: `medarc-eval <ENV> ...`
  - Special rule: the environment name must be the first token.
  - Implemented in `medarc_verifiers/cli/_single_run.py`.
- **Batch mode**: `medarc-eval bench --config <yaml>`
  - Loads config, expands job matrix, creates/updates a run manifest, then executes jobs.
  - Implemented across:
    - Config loading + matrix expansion: `medarc_verifiers/cli/_config_loader.py`
    - Schemas: `medarc_verifiers/cli/_schemas.py`
    - Job expansion: `medarc_verifiers/cli/_job_builder.py`
    - Manifest creation + conflict detection: `medarc_verifiers/cli/_manifest.py`
    - Resume/restart planning: `medarc_verifiers/cli/_manifest_planner.py`
    - Execution loop: `medarc_verifiers/cli/_job_executor.py`
- **Processing**: `medarc-eval process ...`
  - Pipeline wiring: `medarc_verifiers/cli/process/pipeline.py`
- **Win rates**: `medarc-eval winrate ...`
  - Runner that reads processed datasets and writes results: `medarc_verifiers/cli/winrate/runner.py`
  - Core computations live in `medarc_verifiers/cli/winrate/api.py`.

Shared CLI constants (paths, command strings): `medarc_verifiers/cli/_constants.py`.

### How single-run “dynamic env flags” works

Single-run mode introspects each environment’s `load_environment()` signature (and docstring) to generate argparse flags on the fly:

- Introspection + validation: `medarc_verifiers/cli/utils/env_args.py`

That’s why `medarc-eval longhealth --help` shows environment-specific flags even though they aren’t hardcoded. For anything too complex for flags, both single/batch support:

- `--env-args '{...json...}'`
- `--env-arg key=value` (repeatable; smart type coercion)

Override parsing helper: `medarc_verifiers/cli/utils/overrides.py`.

## Config + override semantics (batch mode)

Batch configs (YAML) validate into pydantic models in `medarc_verifiers/cli/_schemas.py`. After validation:

- Environment matrices expand into multiple env variants (IDs can be formatted) in `medarc_verifiers/cli/_config_loader.py`.
- Jobs expand into concrete “model × env variant” runs in `medarc_verifiers/cli/_job_builder.py`.

### `env_args` precedence

`env_args` are merged in layers. Think “low → high priority”:

1. Environment config `env.env_args` (from `configs/envs/*.yaml`)
2. Model config `model.env_args`
3. Model env-specific override `model.env_overrides[...]` (lookup tries: env id → matrix base id → module)
4. Job-level overrides `job.env_args`
5. CLI overrides (`--env-args` / `--env-arg`) applied later when building `EvalConfig`

The merge is handled by `medarc_verifiers/cli/utils/env_args.py` (with optional metadata validation).

### `sampling_args` precedence and sanitation

`sampling_args` merge from model → job → CLI, and are then sanitized for OpenAI-compatible clients:

- Unknown parameters are moved under `extra_body` so they can be forwarded to compatible servers (e.g., vLLM).
- Sanitizer: `medarc_verifiers/utils/sampling_args.py`
- Merge point: `medarc_verifiers/cli/_eval_builder.py`

## Endpoints and Prime Inference integration

There are two related concepts:

1. **Endpoint registry** (optional): resolves a model alias to an endpoint URL and key env var.
   - Loader + cache: `medarc_verifiers/cli/utils/endpoint_utils.py`
   - CLI default path: `configs/endpoints.toml` (TOML-first, aligned with upstream verifiers)
   - Legacy Python registries remain usable via explicit `--endpoints-path configs/endpoints.py`.
2. **Prime Inference overrides**:
   - Adds `X-Prime-Team-ID` header (if `PRIME_TEAM_ID` is set and base URL is Prime Inference).
   - Optionally injects `extra_body.usage.include = true` for usage reporting.
   - Selects `PRIME_API_KEY` when available for Prime Inference endpoints.
   - Implementation: `medarc_verifiers/utils/prime_inference.py`

Relevant env vars:

- `OPENAI_API_KEY` (default model key var)
- `PRIME_API_KEY`, `PRIME_TEAM_ID` (Prime Inference)
- `MEDARC_INCLUDE_USAGE` (force usage reporting true/false globally)

Programmatic usage (build headers/sampling overrides for a base URL):

```python
from medarc_verifiers.utils.prime_inference import prime_inference_overrides

headers, sampling_overrides, api_key_var = prime_inference_overrides(base_url)
```

### Judge defaults and judge API keys

Judging defaults are centralized and provider-tuned:

- `medarc_verifiers/utils/judge_helpers.py`

Key env vars:

- `JUDGE_API_KEY` (preferred for judge calls)
- fallback to `PRIME_API_KEY` (if judging via Prime Inference) or `OPENAI_API_KEY`.

## Resume, restart, and manifests (batch mode)

Batch mode writes `runs/raw/<run_id>/run_manifest.json` (manifest v3).

- Manifest schema + update methods: `medarc_verifiers/cli/_manifest.py`
- Planning which jobs to run vs reuse: `medarc_verifiers/cli/_manifest_planner.py`

Important concepts:

- A **job** is a resolved combination of model + environment variant + args (plus sampling args).
- Auto-resume tries to find the newest run matching the config checksum and skip completed jobs.
- Restart can “seed” a new run from an old run, reusing outputs when job signatures match.
- Conflict detection is conservative for most fields, but treats some model fields as “resume tolerant” (e.g., base URLs/timeouts) so you can move between providers without being blocked.

## Raw outputs (what eval produces)

Raw outputs are expected under `runs/raw/<run_id>/<job_id>/` and include:

- `results.jsonl`: per-example rollouts
- `summary.json`: aggregated job metrics
- `metadata.json`: job configuration snapshot (env/model/sampling args, etc.)

The runner executes via `verifiers.utils.eval_utils.run_evaluation()` (called from `medarc_verifiers/cli/_single_run.py` and `medarc_verifiers/cli/_job_executor.py`).

## Processing pipeline (raw → parquet)

Docs: `docs/medarc-eval-process.md`.

Entry point: `medarc_verifiers/cli/process/pipeline.py` (via `run_process()`).

### What processing does

1. **Discover** job outputs from `runs/raw` by reading run manifests:
   - `medarc_verifiers/cli/process/discovery.py`
2. **Normalize metadata** by merging manifest fields with `metadata.json`:
   - `medarc_verifiers/cli/process/metadata.py`
3. **Handle rollouts**:
   - Some runs encode rollout indices in env ids like `env-a-rollout7` or `env-a-r7`.
   - If not present, processing can fall back to parsing the results directory name.
   - `medarc_verifiers/cli/process/rollout.py`
4. **Load rows from `results.jsonl`**:
   - Drops large fields (`prompt`, `completion`) by default.
   - Allows selecting extra per-env columns into a JSON-encoded `extras` column.
   - If the JSONL contains multiple rollouts per `example_id`, computes a data-driven `rollout_index` based on occurrence count.
   - Flattens `token_usage` into explicit columns like `model_token_total`, `judge_cost`, etc.
   - `medarc_verifiers/cli/process/rows.py`
5. **Aggregate** rows per `(model_id, base_env_id)` and union schemas:
   - `medarc_verifiers/cli/process/aggregate.py`
6. **Write Parquet**:
   - Output path is `<processed_dir>/<model_id>/<env_id>.parquet`.
   - Adds exporter metadata under a Parquet schema metadata key.
   - Writes `env_index.json` (v2) and `dataset_infos.json` for HF datasets UX.
   - `medarc_verifiers/cli/process/writer.py`, `medarc_verifiers/cli/process/env_index.py`

### Delta processing and HF baselines

Processing can use `env_index.json` to do incremental updates (delta processing). It also supports pulling/pushing processed artifacts to/from Hugging Face:

- HF baseline management (download/copy policies): `medarc_verifiers/cli/process/workspace.py`
- HF sync operations: `medarc_verifiers/cli/hf/sync.py`

## Win rates (processed parquet → comparisons)

Docs: `docs/medarc-eval-winrate.md`.

`medarc-eval winrate` reads dataset inventory from `env_index.json`, then computes pairwise model comparisons.

- Dataset discovery via `env_index.json`: `medarc_verifiers/cli/winrate/runner.py`
- Core math + weighting policies: `medarc_verifiers/cli/winrate/api.py`
- Outputs:
  - timestamped `winrates-<timestamp>.json` and `.csv`
  - `latest.json` and `latest.csv`

## Shared building blocks used by environments

These utilities are frequently imported by environment packages under `environments/*`:

- Prompts and answer format constants: `medarc_verifiers/prompts.py`
- Parsers:
  - XML parser (supports raw string or chat messages): `medarc_verifiers/parsers/xml_parser.py`
  - JSON parser (field alternatives, optional pydantic schema validation, “format reward”): `medarc_verifiers/parsers/json_parser.py`
- Rewards:
  - Robust MCQ grading with CoT/anchored patterns + answer-text fallback: `medarc_verifiers/rewards/multiple_choice_accuracy.py`
  - Normalize judge dimension scores (1–5 → 0–1): `medarc_verifiers/rewards/normalize_helm_reward.py`
- MCQ shuffling with deterministic seeding and “anchor option” preservation:
  - Skips shuffling entirely if options reference other labels (“A or B”, “Both A and C”), to avoid corrupting the question.
  - `medarc_verifiers/utils/randomize_multiple_choice.py`

## Judging and multi-judge support

Some environments use “LLM-as-judge” scoring. `medarc_verifiers` provides:

- A safer judge call wrapper with clearer errors: `medarc_verifiers/judging/judge_core.py`
- A `MultiJudge` that runs multiple judge models concurrently: `medarc_verifiers/judging/multi_judge.py`
- A `verifiers`-compatible rubric wrapper: `medarc_verifiers/judging/multi_judge_rubric.py`

## vLLM orchestrator (local Docker) – separate CLI

Docs: `docs/medarc-orchestrate.md`.

This is a separate tool (`medarc-orchestrate`) for running batch configs against locally hosted vLLM Docker containers with GPU/port scheduling.

- CLI entry: `medarc_verifiers/orchestrate/cli.py`
- Runtime loop: `medarc_verifiers/orchestrate/run.py`

It essentially:

1. Launches vLLM containers
2. Waits for readiness
3. Runs `uv run medarc-eval bench --config ... --api-base-url <allocated>`
4. Tracks orchestration state under `outputs/orchestrator/<run_id>/`

## Where to change things (quick mental index)

- Add/adjust CLI flags or command behavior:
  - `medarc_verifiers/cli/main.py`, `medarc_verifiers/cli/_single_run.py`
- Change config semantics (matrix, normalization, validation):
  - `medarc_verifiers/cli/_config_loader.py`, `medarc_verifiers/cli/_schemas.py`
- Fix resume/restart quirks:
  - `medarc_verifiers/cli/_manifest.py`, `medarc_verifiers/cli/_manifest_planner.py`
- Add new columns or modify processed dataset schema:
  - extraction: `medarc_verifiers/cli/process/rows.py`
  - allowed columns/output schema: `medarc_verifiers/cli/process/writer.py`
- Change winrate math/output:
  - `medarc_verifiers/cli/winrate/api.py`, `medarc_verifiers/cli/winrate/runner.py`
- Adjust judging defaults/provider behaviors:
  - `medarc_verifiers/utils/judge_helpers.py`, `medarc_verifiers/utils/prime_inference.py`
