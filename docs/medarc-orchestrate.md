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
suite = "medmarks-verified.toml"
endpoints_path = "medmarks-endpoints.toml"
eval_images_config = "eval_images.toml"
output_dir = "../outputs/orchestrate/qwen3_5-4b-medmarks-verified"
readiness_timeout_s = 1800
prune_logs_on_success = true

[container]
volumes = ["/path/to/hf-cache:/root/.cache/huggingface"]

[bench]
max_concurrent = 768

[[target]]
endpoint_id = "qwen3.5-4b-instruct"

[[target]]
endpoint_id = "qwen3.5-4b-thinking"
```

Small experiments can use shorthand:

```bash
uv run medarc-orchestrate run --suite configs/longhealth-smoke.toml --endpoint qwen3.5-4b-instruct
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
endpoint_id = "qwen3.5-4b-instruct"
model = "Qwen/Qwen3.5-4B"
api_client_type = "openai_chat_completions"

[endpoint.orchestrate.vllm]
gpus = 1

[endpoint.orchestrate.container]
image = "vllm/vllm-openai:latest"
```

Container mounts that are specific to a launch belong in plan `[container]`, not endpoint metadata. Slurm policy such as account, partition, qos, and nice comes from explicit CLI flags or `[endpoint.orchestrate.slurm]`. If construct image materialization is enabled, the container image must use a pinned non-`latest` tag or digest, or an existing absolute `.sqsh` path.

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

### Construct and Teardown Jobs

Plans may opt into CPU-only lifecycle jobs around each GPU eval task. The construct phase can prefetch Hugging Face model weights into the cache mounted into the vLLM container and materialize the vLLM OCI image into a deterministic Enroot/Pyxis `.sqsh` image. The eval job then depends on construct with `afterok`; optional teardown depends on eval with `afterany`.

```toml
[construct]
enabled = true
cpus = 8
time = "02:00:00"
partition = "cpu"
prefetch_model_weights = true
materialize_images = true

[construct.cache]
# Optional when inferred from a /root/.cache/huggingface volume.
hf_home = "/path/to/hf-cache"
hub_cache = "/path/to/hf-cache/hub"
image_dir = "/path/to/pyxis-images/vllm"
latest_link = true

[teardown]
enabled = false
remove_model_weights = false
remove_images = false
```

When `[container].volumes` includes `/path/to/hf-cache:/root/.cache/huggingface`, construct infers `hf_home = "/path/to/hf-cache"` and `hub_cache = "/path/to/hf-cache/hub"`. The Pyxis worker receives the matching container-side cache env vars: `HF_HOME=/root/.cache/huggingface` and `HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub`.

Image materialization uses direct `enroot import` and writes the configured deterministic `.sqsh` path before the GPU job starts. It requires `[construct.cache].image_dir` for OCI images and rejects mutable `:latest` images. Existing absolute `.sqsh` image paths are treated as already materialized and are left unchanged.

Teardown deletion is intentionally conservative. Model-weight deletion is only for isolated per-run cache roots; shared production caches should leave teardown disabled and rely on a separate retention policy. For preemptible idle-capacity jobs, use Slurm requeue for the eval job. The teardown `afterany` dependency is expected to release only after the same requeued eval job id reaches final completion.

### Task Bundles

Before launching a task, the orchestrator creates a bundle under `outputs/orchestrate/<run_id>/tasks/<task-slug>/`:

- `eval-config.toml`: generated task-local bench config.
- `task.yaml`: resolved worker spec.
- `orchestrate-snapshot.toml`: matched endpoint runtime entry and registry provenance.
- `eval_images-snapshot.toml`: selected eval images and registry provenance.
- `allocation.json`: GPU/port allocation for the worker.
- `construct.sh` / `teardown.sh`: CPU-only lifecycle scripts when enabled.
- `runtime/construct_result.json` / `runtime/teardown_result.json`: lifecycle result artifacts when enabled.
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
