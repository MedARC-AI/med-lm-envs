# medarc-verifiers

Utilities and CLI for running medical LLM benchmarks with [verifiers](https://github.com/primeintellect-ai/verifiers). Provides TOML bench execution, result processing, and shared building blocks for authoring environments.

## Install

```bash
pip install medarc-verifiers
```

Environments are installed separately via `prime env install <owner/env>` (from the [Prime Intellect Hub](https://app.primeintellect.ai)) or `vf-install <env>` (from a local directory).

## medarc-eval

`medarc-eval` covers the full evaluation pipeline:

| Command | Description |
|---------|-------------|
| `medarc-eval <ENV>` | Run a single benchmark; env-specific flags inferred from `load_environment()` |
| `medarc-eval bench` | Run upstream TOML eval configs with deterministic MedARC paths |
| `medarc-eval process` | Convert raw outputs to analysis-ready parquet |
| `medarc-eval winrate` | Compute HELM-style win rates across models |

See [medarc-eval.md](medarc-eval.md) for full documentation.
