## vLLM Orchestrator

`medarc-orchestrate` submits eval suites to Slurm/Pyxis vLLM jobs. The suite is ordinary `medarc-eval bench` TOML without a model selection; the launch plan chooses endpoint targets and the orchestrator generates task-local bench configs and readable Slurm scripts.

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
bundle_dir = "../outputs/orchestrate/qwen3_5-4b-medmarks-verified"
output_dir = "../runs/verified/qwen3_5-4b-medmarks-verified"
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

### Auxiliary Images

`eval_images_config` selects auxiliary Pyxis images by eval or env id. These images are rendered directly into each task's `submit.sh` and launch inside the same Slurm allocation before `medarc-orchestrate launch` starts vLLM. Use `{port}` in an auxiliary image command, env, readiness URL, or injected env arg when the image needs a per-job node-local port.

```toml
[[eval_image]]
id = "medagentbench-fhir"
envs = ["medagentbench", "medagentbenchv2"]
runtime = "pyxis"
image = "/path/to/medagentbench_withsh.sqsh"
srun_args = ["--mem=16G", "--no-container-entrypoint", "--container-env=JAVA_TOOL_OPTIONS,SERVER_PORT"]
command = ["/usr/bin/java", "--class-path", "/app/main.war", "org.springframework.boot.loader.PropertiesLauncher"]

[eval_image.env]
JAVA_TOOL_OPTIONS = "-XX:+UseSerialGC -Xms256m -Xmx1024m"
SERVER_PORT = "{port}"

[eval_image.readiness]
url = "http://127.0.0.1:{port}/fhir/metadata"
timeout_s = 240

[eval_image.inject.env_args]
fhir_api_base = "http://127.0.0.1:{port}/fhir/"
```

### Generated Eval Configs

Each task bundle gets `<task>/eval-config.toml`. It contains the suite's `[[eval]]` and `[[ablation]]` entries plus resolved top-level values:

- `endpoint_id`
- `endpoints_path`
- `output_dir`, from plan `output_dir`, otherwise the suite's `output_dir`, otherwise `<task>/bench`
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

Container mounts that are specific to a launch belong in plan `[container]`, not endpoint metadata. Slurm policy such as account, partition, qos, and nice comes from explicit CLI flags or `[endpoint.orchestrate.slurm]`. If prepare image materialization is enabled, mutable image tags such as `:latest` are resolved to immutable registry digests before import.

The vLLM launch environment defaults `SAFETENSORS_FAST_GPU=1`. When `[container].volumes` mounts a host cache to `/root/.cache/huggingface`, the worker also sets container-side `HF_HOME=/root/.cache/huggingface` and `HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub` automatically so Hugging Face uses the mounted cache instead of the user's home cache. Use `[container].env_file` only when you need to override those defaults or pass additional container environment:

```dotenv
HF_HOME=/root/.cache/huggingface
HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
SAFETENSORS_FAST_GPU=1
```

### Slurm Usage

Slurm options come from CLI overrides first, then `[endpoint.orchestrate.slurm]`, then built-in defaults. GPU allocation is derived per task from `[endpoint.orchestrate.vllm].gpus`; there is no Python-side concurrency throttle.
The built-in Slurm defaults use `qos = "bottom"` and leave `nice` unset so the cluster default applies unless a plan, endpoint entry, or CLI flag sets it explicitly.

Common flags:

- `--eval-images-config` overrides the eval image registry.
- `--endpoints-path` selects the endpoint registry used for target resolution and bench.
- `--bundle-dir` sets the orchestrator task bundle root.
- `--output-dir` sets the `medarc-eval bench` result root.
- `--account`, `--partition`, `--qos`, `--nice`, and `--time` explicitly override Slurm submission settings.
- `--dependency` applies an sbatch dependency to submitted tasks.
- `--prune-logs-on-success` removes per-task serve and bench logs after successful tasks.

Status uses `slurm_manifest.json` to find the prepared/eval/teardown job ids, then queries Slurm for live state and accounting details. The JSON output includes the commands run, current `squeue`/`scontrol` fields when available, `sacct --duplicates` attempts, restart counts, and per-task live Slurm fields such as `eval_slurm_live_state`, `eval_slurm_reason`, `eval_slurm_restarts`, and `eval_slurm_preemptions`. Runtime summary files are still included as task-local context, but Slurm is the source of truth for current queue state.

```bash
uv run medarc-orchestrate status --run-id qwen-run
uv run medarc-orchestrate status --bundle-dir outputs/orchestrate/qwen-run --json
```

### Prepare and Teardown Jobs

Plans may opt into CPU-only lifecycle jobs around each GPU eval task. The prepare phase can prefetch Hugging Face model weights into the cache mounted into the vLLM container and materialize the vLLM OCI image into a deterministic Enroot/Pyxis `.sqsh` image. The eval job then depends on prepare with `afterok`; optional teardown depends on eval with `afterany`.

```toml
[prepare]
enabled = true
cpus = 8
time = "02:00:00"
partition = "main"
prefetch_model_weights = true
materialize_images = true

[prepare.cache]
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

When `[container].volumes` includes `/path/to/hf-cache:/root/.cache/huggingface`, prepare infers `hf_home = "/path/to/hf-cache"` and `hub_cache = "/path/to/hf-cache/hub"`. The GPU eval worker infers the matching container-side env from the same mount and passes `HF_HOME` and `HUGGINGFACE_HUB_CACHE` through Pyxis. If vLLM logs show Hugging Face writing under a user home directory, check that the rendered eval job includes the canonical `/root/.cache/huggingface` mount.

Image materialization uses direct `enroot import` before the GPU job starts. For regular tags or digest-pinned images, prepare writes a deterministic `.sqsh` path under `[prepare.cache].image_dir`. For mutable tags such as `:latest`, prepare first queries the registry, resolves the tag to a digest, imports the digest-specific image if needed, and atomically updates `latest.sqsh` and `latest` symlinks to that image. Existing absolute `.sqsh` image paths are treated as already materialized and are left unchanged.

Teardown deletion is intentionally scoped. Model-weight deletion removes only the matching Hugging Face repo directory under the configured hub cache, while shared production image caches should keep `remove_images = false` and rely on a separate image retention policy. For preemptible idle-capacity jobs, use Slurm requeue for the eval job. Eval jobs default to `#SBATCH --signal=B:TERM@120`; the generated batch script forwards that signal to `medarc-orchestrate launch`, which terminates the active `medarc-eval bench` subprocess and records a cancelled runtime state before Slurm requeues or cancels the allocation. Override with `[endpoint.orchestrate.slurm] signal = "B:TERM@60"` or `medarc-orchestrate run --slurm-signal B:TERM@60` when a cluster needs a different grace window. The teardown `afterany` dependency is expected to release only after the same requeued eval job id reaches final completion.

### Task Bundles

Before launching a task, the orchestrator creates a bundle under `outputs/orchestrate/<run_id>/tasks/<task-slug>/`:

- `eval-config.toml`: generated task-local bench config.
- `prepare.sh`: optional CPU-only cache/image preparation script.
- `submit.sh`: GPU script that starts auxiliary images, launches vLLM, and runs `medarc-eval bench`.
- `teardown.sh`: optional CPU-only cleanup script.
- `orchestrate-snapshot.toml`: matched endpoint runtime entry and registry provenance.
- `eval_images-snapshot.toml`: selected eval images and registry provenance.
- `runtime/prepare_result.json` / `runtime/teardown_result.json`: lifecycle result artifacts when enabled.
- `runtime/`: runtime state, result, and summary files.
- `serve/`: vLLM logs and serve-side runtime files.
- `bench/`: task-local benchmark stdout/stderr and fallback `medarc-eval bench --output-dir` root when no plan or suite `output_dir` is set.

Task bundles do not contain `task.yaml`, `allocation.json`, `construct-allocation.json`, or `teardown-allocation.json`. The generated scripts carry the explicit command arguments instead.

`submit.sh` runs:

```bash
medarc-orchestrate launch <explicit vLLM/container/port flags> -- \
  medarc-eval bench --config <task>/eval-config.toml --api-base-url <local-url> --provider local --output-dir <output-dir>
```

### Processing Outputs

Process orchestrated outputs by pointing `medarc-eval process` at the configured result root. If no plan or suite `output_dir` was set, use the orchestrator run root so discovery finds task-local fallback `bench/` outputs:

```bash
uv run medarc-eval process --runs-dir outputs/orchestrate/<run_id> --output-dir runs/processed
```
