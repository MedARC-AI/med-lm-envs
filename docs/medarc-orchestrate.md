## vLLM Orchestrator

`medarc-orchestrate` submits eval suites to Slurm/Pyxis vLLM workers. The suite is ordinary `medarc-eval bench` TOML without a model selection; the launch plan chooses endpoint targets and the orchestrator generates task-local bench configs.

### Concepts

- Suite: benchmark definitions such as `configs/medmarks-verified.toml`.
- Target: one endpoint id to run against a suite.
- Endpoint registry: model/runtime metadata under `[[endpoint]]` and `[endpoint.orchestrate]`.
- Plan: launch recipe tying suites, targets, registries, and deployment overrides together.

### Plan Example

```toml
name = "qwen3_5-4b-medmarks-verified"
suite = "configs/medmarks-verified.toml"
endpoints_path = "configs/medmarks-endpoints.toml"
eval_images_config = "configs/eval_images.toml"
output_dir = "outputs/orchestrate/qwen3_5-4b-medmarks-verified"
readiness_timeout_s = 1800
prune_logs_on_success = true

[container]
volumes = ["/data/medlm_cache:/root/.cache/huggingface"]

[bench]
max_concurrent = 768

[[target]]
endpoint_id = "qwen3_5-4b-instruct"

[[target]]
endpoint_id = "qwen3_5-4b-thinking"
```

Small experiments can use shorthand:

```bash
uv run medarc-orchestrate run --suite configs/longhealth-smoke.toml --endpoint qwen3_5-4b-instruct
```

The canonical launch path is:

```bash
uv run medarc-orchestrate run --plan configs/qwen3_5-4b-medmarks-verified-plan.toml --dry-run
uv run medarc-orchestrate run --plan configs/qwen3_5-4b-medmarks-verified-plan.toml
```

### Generated Eval Configs

Each task bundle gets `<task>/eval-config.toml`. It contains the suite's `[[eval]]` and `[[ablation]]` entries plus orchestrator-owned top-level values:

- `endpoint_id`
- `endpoints_path`
- `output_dir = "<task>/bench"`
- typed `[bench]` defaults from the plan/target

Suites may set normal bench defaults, but must not set `endpoint_id`, `model`, or `endpoints_path`.

### Runtime Metadata

Runtime settings live on endpoint registry entries:

```toml
[[endpoint]]
endpoint_id = "qwen3_5-4b-instruct"
model = "Qwen/Qwen3.5-4B-Instruct"
api_client_type = "openai_chat_completions"

[endpoint.orchestrate.vllm]
gpus = 1

[endpoint.orchestrate.container]
image = "vllm/vllm-openai:latest"
```

Container mounts that are specific to a launch belong in plan `[container]`, not endpoint metadata. Slurm policy such as account, partition, qos, and nice comes from explicit CLI flags or `[endpoint.orchestrate.slurm]`.

### Slurm Usage

Slurm options come from CLI overrides first, then `[endpoint.orchestrate.slurm]`, then built-in defaults. GPU allocation is derived per task from `[endpoint.orchestrate.vllm].gpus`; there is no Python-side concurrency throttle.

Common flags:

- `--eval-images-config` overrides the eval image registry.
- `--endpoints-path` selects the endpoint registry used for target resolution and bench.
- `--output-dir` sets the orchestrator output root.
- `--account`, `--partition`, `--qos`, `--nice`, and `--time` explicitly override Slurm submission settings.
- `--dependency` applies an sbatch dependency to submitted tasks.
- `--prune-logs-on-success` removes per-task serve and bench logs after successful tasks.

Status reads the Slurm submission manifest and worker summary when present:

```bash
uv run medarc-orchestrate status --run-id qwen-run
uv run medarc-orchestrate status --output-dir outputs/orchestrate/qwen-run --json
```

### Task Bundles

Before launching a task, the orchestrator creates a bundle under `outputs/orchestrate/<run_id>/tasks/<task-slug>/`:

- `eval-config.toml`: generated task-local bench config.
- `task.yaml`: resolved worker spec.
- `orchestrate-snapshot.toml`: matched endpoint runtime entry and registry provenance.
- `eval_images-snapshot.toml`: selected eval images and registry provenance.
- `allocation.json`: GPU/port allocation for the worker.
- `bench/`: task-local `medarc-eval bench --output-dir` root.
- `serve/` and `runtime/`: runtime logs, state, and task manifest files.

Workers run:

```bash
medarc-eval bench --config <task>/eval-config.toml --api-base-url <local-url> --provider local --output-dir <task>/bench
```

### Processing Outputs

Process orchestrated outputs by pointing `medarc-eval process` at the orchestrator run root or a parent directory. Discovery recursively finds nested `results.jsonl` and `metadata.json` files under task-local `bench/` directories:

```bash
uv run medarc-eval process --runs-dir outputs/orchestrate/<run_id> --output-dir runs/processed
```
