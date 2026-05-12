# MedARC Eval TOML Configs

These configs use upstream `verifiers` TOML semantics. Repeated `env_id` entries
and `[[ablation]]` sweeps intentionally keep the upstream environment id stable;
`medarc-eval bench` writes deterministic variant directories for differing
`env_args` and `sampling_args`.

```bash
medarc-eval bench --config configs/medmarks-smoke.toml --dry-run
medarc-eval bench --config configs/medmarks-verified.toml
medarc-eval process --runs-dir runs/evals --output-dir runs/processed
```

Use `medmarks-endpoints.toml` when you want one of the Medmarks model aliases
and its sampling defaults:

```bash
medarc-eval bench \
  --config configs/medmarks-verified.toml \
  --endpoints-path configs/medmarks-endpoints.toml \
  -m gpt-oss-20b-low \
  --api-base-url https://api.pinference.ai/api/v1 \
  --api-key-var PRIME_API_KEY \
  --dry-run
```

`medmarks-endpoints.toml` is a portable alias registry. It maps endpoint IDs to
model IDs, client types, and sampling defaults, but intentionally omits `url`,
`key`, and `max_concurrent` because those are deployment-specific. Supply those
settings with `--provider` or with `--api-base-url` and `--api-key-var`.
The gpt-oss aliases use the Verifiers `openai_responses` client type.

For a local vLLM server exposing an OpenAI-compatible API, keep using the same
alias registry and override only the deployment settings:

```bash
VLLM_API_KEY=local-key medarc-eval bench \
  --config configs/medmarks-verified.toml \
  --endpoints-path configs/medmarks-endpoints.toml \
  -m gpt-oss-20b-low \
  --api-base-url http://127.0.0.1:8000/v1 \
  --api-key-var VLLM_API_KEY \
  --dry-run
```

Per-environment `[tool.verifiers.eval]` defaults are read from editable installs
where the environment `pyproject.toml` is discoverable next to the module. Wheel
installs may ignore those defaults unless the package includes `pyproject.toml`,
so production suite configs keep explicit `num_examples` and
`rollouts_per_example` values.
