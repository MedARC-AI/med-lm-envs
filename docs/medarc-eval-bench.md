# TOML Bench Mode

`medarc-eval bench` runs upstream `verifiers` TOML eval configs sequentially with
MedARC-specific deterministic output paths. It is the supported path for
systematic local benchmark runs.

The old MedARC YAML benchmark runner has been removed. `bench --config` now
accepts `.toml` files only.

## Quick Start

```bash
# Preview the repository smoke config
medarc-eval bench --config configs/eval/smoke.toml --dry-run

# Run the verified production suite
medarc-eval bench --config configs/eval/medmarks-verified.toml

# Run the verified suite against a local OpenAI-compatible server
medarc-eval bench \
  --config configs/eval/medmarks-verified.toml \
  --api-base-url http://127.0.0.1:8000/v1 \
  --provider local \
  --model openai/my-local-model
```

Repository suite configs live in `configs/eval/`:

| Config | Purpose |
|--------|---------|
| `smoke.toml` | Small smoke test used by CLI tests |
| `medmarks-verified.toml` | Verified benchmark suite |
| `medmarks-open_ended.toml` | Open-ended benchmark suite |

## Config Format

Bench configs use upstream `verifiers` TOML semantics: top-level defaults plus
one or more `[[eval]]` blocks. MedARC adds deterministic output planning around
the resolved evals and writes a required `bench_index.json` sidecar; it does not
use YAML `models`, `envs`, or `jobs` sections.

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
num_examples = -1
rollouts_per_example = 1
env_args = { shuffle_answers = true }

[ablation.sweep.env_args]
shuffle_seed = [1618, 9331]
```

Example output paths:

```text
runs/evals/openai-gpt-4.1-mini/medqa/env_args.shuffle_seed-1618/
runs/evals/openai-gpt-4.1-mini/medqa/env_args.shuffle_seed-9331/
```

Non-variant evals write to `runs/evals/<model>/<env>/`.

## Output Sidecar

Every bench output root contains `bench_index.json`. It records one entry for
each successfully materialized eval output, including `results_path`, `model`,
`env_id`, optional variant identity, resolved args, and a `plan_digest`.

Processing prefers this sidecar over path inference for bench outputs. Failed
evals are omitted from the sidecar until their `metadata.json` and
`results.jsonl` exist, so `--continue-on-error` runs leave successful siblings
processable.

## Resume and Force

Bench writes each eval to a deterministic result directory. Re-running the same
TOML config reuses the same directory only when the matching `bench_index.json`
entry has the same `plan_digest` and existing `metadata.json` contains matching
MedARC identity fields.

```bash
# Resume matching deterministic outputs
medarc-eval bench --config configs/eval/medmarks-verified.toml

# Archive existing deterministic outputs and rerun
medarc-eval bench --config configs/eval/medmarks-verified.toml --force
```

The `plan_digest` and MedARC metadata identity payload are based on canonical
JSON for the planned eval identity, including unknown `sampling_args`. MedARC
does not maintain a sampling-argument allowlist for resume safety; new provider
arguments pass through to upstream.

## Common Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Required path to an upstream TOML eval config |
| `--dry-run` | Resolve evals and print the deterministic plan |
| `--force` | Archive existing deterministic output and rerun |
| `--output-dir PATH` | Override the config output directory |
| `--env-dir PATH` | Directory containing local environments |
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
  --config configs/eval/medmarks-verified.toml \
  --api-base-url https://api.pinference.ai/api/v1
```

## Processing Outputs

After a TOML bench run, process the deterministic eval outputs:

```bash
medarc-eval process --runs-dir runs/evals --output-dir runs/processed
medarc-eval winrate --processed-dir runs/processed
```

Processing still supports legacy `runs/raw/<run_id>/run_manifest.json` outputs
for migration, but new bench runs should use `runs/evals`.

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

Old raw outputs remain processable through the legacy manifest reader, so
historical runs do not need to be converted before processing.
