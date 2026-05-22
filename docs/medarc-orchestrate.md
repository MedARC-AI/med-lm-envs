## vLLM Orchestrator

`medarc-orchestrate` runs TOML `medarc-eval bench` configs against locally served vLLM instances. It keeps benchmark semantics in upstream eval TOML files and resolves runtime infrastructure from endpoint registry entries.

### Config Files

A plan can be TOML, YAML, or JSON. Most users only need job configs plus the endpoint registry:

```toml
name = "local-vllm"
job_configs = ["configs/qwen-30b-a3b-medqa.toml"]
eval_images_config = "configs/eval_images.toml"
endpoints_path = "configs/medmarks-endpoints.toml"
env_file = ".env"
gpu_range = "0-3"
port_range = "8000-8999"
max_parallel = 2
readiness_timeout_s = 1800
prune_logs_on_success = true
```

Each `job_configs` entry must be an upstream eval TOML config accepted by `medarc-eval bench`:

```toml
model = "Qwen/Qwen3-30B-A3B"
output_dir = "runs/evals"

[[eval]]
env_id = "medqa"
num_examples = 5
rollouts_per_example = 1
```

Runtime settings live on the matching `[[endpoint]]` entry under `endpoint.orchestrate`. The orchestrator matches by exact
`endpoint_id`; fuzzy matching against a separate model registry is not supported.

```toml
[[endpoint]]
endpoint_id = "qwen-30b-a3b-thinking"
model = "Qwen/Qwen3-30B-A3B-Thinking-2507"
api_client_type = "openai_chat_completions"

[endpoint.sampling_args]
temperature = 0.6
top_p = 0.95

[endpoint.orchestrate.vllm]
gpus = 2

[endpoint.orchestrate.vllm.serve]
max_model_len = 40960

[endpoint.orchestrate.slurm]
account = "training"
time = "04:00:00"
```

The orchestrator defaults `tensor_parallel_size` to `gpus`. It also supplies default container values for `image`,
`container_port`, and `ipc_mode`, plus default Slurm values for `qos`, `nice`, and `slurm_resume`. It supplies default Pyxis
`srun_extra_args` and default vLLM
serve values for `gpu_memory_utilization`, `max_model_len`, `async_scheduling`, `enable_prefix_caching`, and
`enable_auto_tool_choice`. Set those keys only when a model needs to override the built-in defaults.

Eval auxiliary images, such as benchmark services, live in `eval_images.toml` and are selected by eval or env id:

```toml
[[eval_image]]
id = "medagentbench-fhir"
evals = ["medagentbenchv2_patient", "medagentbenchv2_test"]
runtime = "pyxis"
image = "/path/to/medagentbench_withsh.sqsh"
command = ["bash", "-lc", "serve-fhir"]

[eval_image.readiness]
url = "http://127.0.0.1:8080/health"
timeout_s = 240
```

### Local Usage

```bash
uv run medarc-orchestrate run --plan plans/local-vllm.toml --runtime podman
uv run medarc-orchestrate run --job-config configs/qwen-30b-a3b-medqa.toml --endpoints-path configs/medmarks-endpoints.toml --runtime pyxis
```

Common flags:

- `--runtime {docker,podman,pyxis}` selects the serve backend.
- `--eval-images-config` overrides the eval image registry.
- `--endpoints-path` selects the endpoint registry used for endpoint/model resolution, orchestration settings, and bench.
- `--output-dir` sets the orchestrator output root.
- `--max-parallel`, `--gpu-range`, and `--port-range` control local scheduling.
- `--prune-logs-on-success` removes per-task serve and bench logs after successful tasks.

### Slurm Usage

```bash
uv run medarc-orchestrate run --backend slurm --plan plan-qwen-small-slurm.toml --dry-run
uv run medarc-orchestrate run --backend slurm --plan plan-qwen-small-slurm.toml --output-dir outputs/orchestrate/qwen-run
```

Slurm options come from `[endpoint.orchestrate.slurm]`, with Slurm executor CLI overrides taking precedence.
`slurm_resume = true` renders `#SBATCH --requeue`, so resubmitting the same task bundle reuses the same task-local
bench output directory. The retained `medarc-orchestrate slurm` command is a thin alias for the same launch resolver.

### Task Bundles

Before launching a task, the orchestrator creates a task bundle under `outputs/orchestrate/<run_id>/tasks/<task-slug>/`:

- `eval-config.toml`: copied eval TOML used by the worker.
- `task.yaml`: resolved task spec and registry snapshots for worker execution.
- `orchestrate-snapshot.toml`: matched model runtime entry and registry provenance.
- `eval_images-snapshot.toml`: selected eval images and registry provenance.
- `allocation.json`: GPU/port allocation for the worker.
- `bench/`: deterministic `medarc-eval bench --output-dir` root.
- `serve/` and `runtime/`: runtime logs, state, and task manifest files.

The worker always runs bench against bundled `eval-config.toml`, not the original source path:

```bash
medarc-eval bench --config <task>/eval-config.toml --api-base-url <local-url> --provider local --output-dir <task>/bench
```

Removed YAML-runner flags such as `--run-id`, `--restart`, and `--on-complete` are not passed to bench. Requeue and retry behavior relies on TOML bench deterministic output paths.

### Processing Outputs

Process orchestrated task outputs by pointing `medarc-eval process` at the orchestrator run root or a parent directory. Discovery recursively finds nested `results.jsonl` and `metadata.json` files under task-local `bench/` directories:

```bash
uv run medarc-eval process --runs-dir outputs/orchestrate/<run_id> --output-dir runs/processed
```

Metadata remains authoritative for model and environment identity. The orchestrator does not add a separate manifest-based processing path.
