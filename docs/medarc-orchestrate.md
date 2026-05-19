## vLLM Orchestrator

`medarc-orchestrate` runs TOML `medarc-eval bench` configs against locally served vLLM instances. It keeps benchmark semantics in upstream eval TOML files and resolves runtime infrastructure from separate registries.

### Config Files

A plan is still YAML because it is an orchestrator control file, not a bench config:

```yaml
name: local-vllm
job_configs:
  - configs/qwen-30b-a3b-medqa.toml
orchestrate_config: configs/orchestrate.toml
eval_images_config: configs/eval_images.toml
endpoints_path: configs/endpoints.toml
env_file: .env
gpu_range: "0-3"
port_range: "8000-8999"
max_parallel: 2
readiness_timeout_s: 1800
prune_logs_on_success: true
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

Runtime settings live in `orchestrate.toml`:

```toml
schema_version = 1

[[model]]
id = "Qwen/Qwen3-30B-A3B"
aliases = ["qwen-30b-a3b"]

[model.vllm]
gpus = 2
tensor_parallel_size = 2

[model.vllm.serve]
max_model_len = 40960

[model.container]
image = "vllm/vllm-openai:latest"
container_port = 8000
volumes = ["/data/huggingface:/root/.cache/huggingface:rw"]
ipc_mode = "host"

[model.pyxis]
srun_extra_args = ["--overlap"]

[model.slurm]
account = "training"
time = "04:00:00"
slurm_resume = true
```

Eval auxiliary images, such as benchmark services, live in `eval_images.toml` and are selected by eval or env id:

```toml
schema_version = 1

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
uv run medarc-orchestrate local --plan plans/local-vllm.yaml --runtime podman
uv run medarc-orchestrate local --job-config configs/qwen-30b-a3b-medqa.toml --orchestrate-config configs/orchestrate.toml --runtime pyxis
```

Common flags:

- `--runtime {docker,podman,pyxis}` selects the serve backend.
- `--orchestrate-config` overrides the model runtime registry.
- `--eval-images-config` overrides the eval image registry.
- `--endpoints-path` is used for endpoint/model resolution and is passed through to `medarc-eval bench`.
- `--output-dir` sets the orchestrator output root.
- `--max-parallel`, `--gpu-range`, and `--port-range` control local scheduling.
- `--prune-logs-on-success` removes per-task serve and bench logs after successful tasks.

### Slurm Usage

```bash
uv run medarc-orchestrate slurm --plan plan-qwen-small-slurm.yaml --dry-run
uv run medarc-orchestrate slurm --plan plan-qwen-small-slurm.yaml --output-dir outputs/orchestrate/qwen-run
```

Slurm options come from `[model.slurm]` in `orchestrate.toml`, with CLI overrides taking precedence. `slurm_resume = true` renders `#SBATCH --requeue`, so resubmitting the same task bundle reuses the same task-local bench output directory.

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
