# TOML Bench Mode

`medarc-eval bench` runs upstream `verifiers` TOML eval configs sequentially with
MedARC-specific deterministic output paths. It is the supported path for
systematic local benchmark runs.

The old MedARC YAML benchmark runner has been removed. `bench --config` now
accepts `.toml` files only.

## Quick Start

```bash
# Preview the repository smoke config
medarc-eval bench --config configs/medmarks-smoke.toml --dry-run

# Run the verified production suite
medarc-eval bench --config configs/medmarks-verified.toml

# Require all selected env packages to already be installed
medarc-eval bench --config configs/medmarks-verified.toml --no-auto-install

# Run the verified suite against a local OpenAI-compatible server
medarc-eval bench \
  --config configs/medmarks-verified.toml \
  --api-base-url http://127.0.0.1:8000/v1 \
  --provider local \
  --model openai/my-local-model
```

Repository suite configs live in `configs/`:

| Config | Purpose |
|--------|---------|
| `medmarks-smoke.toml` | Small Medmarks-V smoke test used by CLI tests |
| `medmarks-verified.toml` | Verified benchmark suite |
| `medmarks-open_ended.toml` | Open-ended benchmark suite |

## Config Format

Bench configs use upstream `verifiers` TOML semantics: top-level defaults plus
one or more `[[eval]]` blocks. MedARC adds deterministic output planning around
selected raw eval configs; it does not use YAML `models`, `envs`, or `jobs`
sections.

```toml
model = "openai/gpt-4.1-mini"
save_results = true
output_dir = "runs/evals"

[[eval]]
env_id = "medqa"
num_examples = 25
rollouts_per_example = 1
env_args = { shuffle_answers = true, shuffle_seed = 1618 }
sampling_args = { temperature = 0.0 }

[[eval]]
env_id = "pubmedqa"
num_examples = 25
rollouts_per_example = 1
```

Per-environment defaults can also live in an environment package
`pyproject.toml` under `[tool.verifiers.eval]`. Production suite configs keep
explicit `num_examples` and `rollouts_per_example` values so they remain stable
across editable and wheel installs.

## Local Environment Install Lifecycle

By default, TOML bench auto-installs selected local environment packages that
are not already importable in the active Python environment. Auto-install only
applies to missing local packages resolved from `--env-dir`; selected envs that
are already importable keep the normal in-process execution path.

`--env-dir` defaults to `environments/`. When auto-install is needed, bench
creates a system temporary directory with a `medarc-bench-venv-` prefix, creates
a venv inside it with `uv venv`, installs the selected local env package
editable into that venv, runs one eval through the private bench child, and then
removes the temporary venv.

```bash
medarc-eval bench \
  --config configs/medmarks-verified.toml \
  --eval-index "$SLURM_ARRAY_TASK_ID"
```

When a selected env package is missing, bench prints a warning to stderr and
runs that eval in an isolated temporary venv. The parent process loads and
expands the TOML config, applies `--eval-index` / `--start-at` / `--stop-after`,
plans deterministic output paths from raw TOML and CLI values, creates a temp
venv, installs MedARC into it, installs the target env package into it, runs the
bench child with the parent-planned `resume_path`, and deletes the temp venv.

If the active `medarc-verifiers` install is editable, isolated mode installs
that same checkout editable into the temp venv. If the active install is not
editable, isolated mode installs `medarc-verifiers==<current-version>` and
requires that package/version to be resolvable by the normal package resolver.
If resolution fails, run from an editable checkout or preinstall env packages
and pass `--no-auto-install`.

For faster strict local iteration, preinstall environments and opt out:

```bash
vf-install medqa
vf-install pubmedqa
medarc-eval bench --config configs/medmarks-verified.toml --no-auto-install
```

`--dry-run` does not create venvs, install packages, or spawn child processes.
If selected env packages are missing, dry run says they would be auto-installed.
Dry-run identity and deterministic paths are based on TOML and CLI values only;
environment package `[tool.verifiers.eval]` defaults are execution-time defaults
and do not affect dry-run display or path planning.

Isolated mode removes shared Python package metadata mutation from auto-install,
but it is not full filesystem or side-effect isolation. Concurrent runs can
still collide if they target the same deterministic output directory without
unique selections, output roots, or variants. Hugging Face caches, judge caches,
cwd-relative artifacts, temp files created by environment code, and network/API
side effects can also remain shared.

## Ablations and Variants

Use upstream `[[ablation]]` tables to sweep values. The upstream env id stays
unchanged, and MedARC writes each differing config to a deterministic variant
directory.

```toml
model = "openai/gpt-4.1-mini"
save_results = true
output_dir = "runs/evals"

[[ablation]]
env_id = "medqa"
name = "shuffle_seed-{env_args.shuffle_seed}"
num_examples = -1
rollouts_per_example = 1
env_args = { shuffle_answers = true }

[ablation.sweep.env_args]
shuffle_seed = [1618, 9331]
```

Example output paths:

```text
runs/evals/openai-gpt-4.1-mini/medqa/shuffle_seed-1618/
runs/evals/openai-gpt-4.1-mini/medqa/shuffle_seed-9331/
```

Non-variant evals use the reserved variant id `base` and write to
`runs/evals/<model-or-endpoint>/<env>/base/`. When an eval resolves through an
endpoint registry alias, the endpoint id is used for the path's model component
so endpoint modes that share one served model do not collide. The upstream
served model and `metadata.json["model"]` still use the registry entry's
`model`.

Duplicate `(model-or-endpoint, env)` evals must provide an explicit
`variant_id` or `name`. `name` may use simple templates such as
`shuffle_seed-{env_args.shuffle_seed}` after ablation expansion.

`variant_id` and `name` are path identities. They must already be path-safe:
use only letters, numbers, `.`, `_`, and `-`. For example,
`variant_id = "shuffle_seed-1618"` is valid, while
`variant_id = "shuffle seed = 1618"` fails with a clear error.

## Metadata

Upstream `metadata.json` remains a normal `verifiers` file. MedARC does not
write separate bench metadata. Processing recovers exact model and environment
identity from upstream metadata, and recovers variant identity from the
deterministic path segment.

## Output Root, Resume, and Force

Bench writes each eval to a deterministic result directory. If neither
`--output-dir` nor TOML `output_dir` is set, the output root defaults to
`runs/evals`.

Existing valid outputs resume automatically. This makes Slurm retries
idempotent for a fixed `--eval-index`:

```bash
medarc-eval bench --config configs/medmarks-verified.toml --eval-index "$SLURM_ARRAY_TASK_ID"
```

If the deterministic target already contains both `metadata.json` and
`results.jsonl`, MedARC passes that path to upstream `verifiers` as
`resume_path` and lets upstream resume. If the target exists but is malformed or
partial, bench fails unless `--force` is set:

```bash
# Archive existing deterministic outputs and rerun
medarc-eval bench --config configs/medmarks-verified.toml --force
```

`--resume` is still accepted for compatibility, but deterministic bench outputs
resume automatically when valid artifacts exist. MedARC does not maintain a
sampling-argument allowlist or fingerprint blocker for resume safety. New
provider arguments pass through to upstream.

## Common Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Required path to an upstream TOML eval config |
| `--dry-run` | Resolve evals and print the deterministic plan |
| `--force` | Archive existing deterministic output and rerun |
| `--resume` | Compatibility flag; valid deterministic outputs resume automatically |
| `--output-dir PATH` | Override the config output directory, default `runs/evals` |
| `--env-dir PATH` | Directory containing local environments, default `environments` |
| `--auto-install` / `--no-auto-install` | Auto-install missing local env packages in isolated temp venvs (default) or require selected envs to be preinstalled |
| `--endpoints-path PATH` | Endpoint registry path, default `configs/endpoints.toml` |
| `--api-base-url URL` | Override API base URL for every eval |
| `--api-key-var NAME` | Override API key environment variable |
| `--provider NAME` | Override upstream provider shorthand |
| `--model MODEL` | Override model for every eval |
| `--eval-index N` | Run one resolved eval by 1-based index |
| `--start-at N` / `--stop-after N` | Run a contiguous 1-based eval range |
| `--continue-on-error` | Continue after a failed eval |
| `--env-arg KEY=VALUE` / `--env-args JSON` | Apply environment arg overrides |
| `--sampling-arg KEY=VALUE` / `--sampling-args JSON` | Apply sampling arg overrides |
| `--max-concurrent N` | Override max concurrency for every eval |
| `--timeout SEC` | Override request timeout for every eval |
| `--max-retries N` | Override upstream rollout retries for every eval |
| `--sleep SEC` | Sleep after each eval |

## Endpoint Sampling Profiles

MedARC extends upstream `verifiers` TOML endpoint registries with optional
endpoint-level `sampling_args`. Use these for model/provider defaults and
compatibility knobs, such as vLLM-only parameters. Put benchmark experiment
settings in the eval TOML or CLI overrides.

```toml
[[endpoint]]
endpoint_id = "gpt-oss-20b-low-local"
model = "openai/gpt-oss-20b"
url = "http://host.docker.internal:8010/v1"
key = "VLLM_API_KEY"
api_client_type = "openai_responses"

[endpoint.sampling_args]
temperature = 1.0
top_p = 1.0
top_k = 0
reasoning_effort = "low"

[[endpoint]]
endpoint_id = "another-model"
model = "openai/another-model"
url = "http://host.docker.internal:8011/v1"
key = "VLLM_API_KEY"
```

Inline tables are also supported:

```toml
sampling_args = { temperature = 1.0, top_p = 1.0, top_k = 0, reasoning_effort = "low" }
```

Precedence is: Prime Inference defaults, endpoint `sampling_args`, raw scalar
`temperature` / `max_tokens`, raw TOML `sampling_args`, then CLI
`--sampling-args` / `--sampling-arg`. Unknown OpenAI parameters such as `top_k`
are still moved under `extra_body` after the merge.

After `[endpoint.sampling_args]`, TOML keys remain inside that nested table
until the next table header. Start a new `[[endpoint]]` before defining another
endpoint.

## Prime Inference

When `--api-base-url` or a config points at Prime Inference
(`https://api.pinference.ai/api/v1`), MedARC applies the same Prime helpers used
by single-run mode:

- `PRIME_API_KEY` is preferred when available.
- `X-Prime-Team-ID` is added from `PRIME_TEAM_ID`.
- Usage reporting is enabled unless `MEDARC_INCLUDE_USAGE=false` is set.

```bash
export PRIME_API_KEY=...
export PRIME_TEAM_ID=...

medarc-eval bench \
  --config configs/medmarks-verified.toml \
  --api-base-url https://api.pinference.ai/api/v1
```

## Processing Outputs

After a TOML bench run, process the deterministic eval outputs:

```bash
medarc-eval process --runs-dir runs/evals --output-dir runs/processed
medarc-eval winrate --processed-dir runs/processed
```

Processing reads eval-output directories under `runs/evals`. Legacy
`runs/raw/<run_id>/run_manifest.json` outputs must be converted with
`scripts/convert_legacy_raw_runs.py` before processing. New bench runs should
use `runs/evals`.

## Migrating from the Removed YAML Runner

Move old YAML `models` entries into top-level TOML defaults or explicit
`[[eval]]` blocks. Move old `envs` and matrix variants into repeated `[[eval]]`
blocks or upstream `[[ablation]]` sweeps.

Removed YAML-runner concepts no longer exist in `medarc-eval bench`:

- YAML `models`, `envs`, and `jobs` schemas
- `run_manifest.json` creation for new bench runs
- `--run-id`, `--restart`, `--auto-resume`, `--no-auto-resume`
- `--job-id`, `--forced`, `--on-complete`
- custom YAML job status and manifest planning

Old raw outputs must be converted before processing:

```bash
uv run python scripts/convert_legacy_raw_runs.py \
  --raw-dir runs/raw \
  --output-dir runs/evals \
  --dry-run
```

The converter is an operator migration helper. It does not mutate `runs/raw` and
defaults to dry-run; pass `--no-dry-run` to write converted eval outputs.
