## vLLM orchestrator (Docker, Pyxis, or Slurm-native submission)

This repo includes an experimental vLLM orchestrator for running `medarc-eval` against
locally hosted vLLM containers with GPU/port scheduling.

There are two different Slurm-related workflows:

- `medarc-orchestrate --runtime pyxis ...` runs the orchestrator inside an existing Slurm allocation on a GPU node.
- `medarc-orchestrate slurm ...` is the Slurm-native submitter. It generates and submits one `sbatch` job per orchestrator task.

If you want `medarc-orchestrate` itself to submit Slurm jobs, use `medarc-orchestrate slurm`. Use `--runtime pyxis` only when you are already inside an allocation.

### Requirements

- `medarc-orchestrate` runs `medarc-eval bench` as a local subprocess on the same host.
- Container image available for the selected runtime.

Docker runtime:

- Docker installed with NVIDIA runtime support.
- Local GPUs available (NVML via `nvidia-ml-py`).
- vLLM image available locally or pullable.

Pyxis runtime:

- Slurm with the Pyxis plugin installed.
- Enroot available on compute nodes.
- Run `medarc-orchestrate --runtime pyxis ...` inside an existing Slurm allocation on a GPU node.
- The Pyxis `--container-image` value must match cluster policy:
  - registry-backed URI such as `docker://vllm/vllm-openai:latest`
  - or a pre-staged squashfs image path

Notes for Pyxis:

- Recommended interactive workflow: `salloc --nodes=1 --gpus=4 --time=02:00:00`, then `medarc-orchestrate --runtime pyxis ...`.
- `srun --pty bash` shells can block later `srun` steps on some clusters unless overlap is enabled. Prefer `salloc` or `sbatch`.
- Pulling large registry images at launch can fail on systems with limited local storage. Pre-staging a squashfs image is often more reliable.
- In shell scripts, image strings containing `#` may need escaping, for example `nvcr.io\\#nvidia/pytorch:25.01-py3`.

Slurm-native submission:

- Use `medarc-orchestrate slurm ...` when you want the tool to create and submit `sbatch` jobs for you.
- The Slurm-native path always uses Pyxis for the inner runtime in v1.
- This is the recommended path for queued multi-job Slurm execution.
- Generated `sbatch` scripts activate an environment and invoke `medarc-orchestrate` directly with `--no-uv-run`; they do not use `uv run` by default.

Note: This machine does not have Docker installed, so integration runs are not available here.

### Job configs vs plan files

These inputs are shared by both:

- `medarc-orchestrate ...`
- `medarc-orchestrate slurm ...`

The orchestrator can run directly from one or more job configs:

```bash
medarc-orchestrate --job-config configs/job-gpt-oss-20b.yaml
medarc-orchestrate \
  --job-config configs/job-gpt-oss-20b.yaml \
  --job-config configs/job-qwen-30b-a3b.yaml \
  --runtime pyxis
```

Use a plan YAML only when you want a reusable named bundle or shared defaults:

```yaml
name: local-vllm
job_configs:
  - configs/job-gpt-oss-20b.yaml
env_file: .env
gpu_range: "0-3"
port_range: "8000-8999"
max_parallel: 2
readiness_timeout_s: 1800
resume: false
rerun_failed: false
```

Each job config must define exactly one model under `models:` and include a top-level
`orchestrate:` block with per-model serve settings.

The `env_file` is a dotenv file that is loaded for every Docker launch. If unset and a repo-level `.env` exists,
it is used automatically. You can also override it via `--env-file`.

Optional: set `orchestrate.restart` to reuse completed jobs from a previous `medarc-eval` run (it is forwarded as
`medarc-eval bench --restart ...`).

Optional: add a top-level `slurm:` block when using the Slurm-native submitter:

```yaml
slurm:
  job_name: qwen-30b-a3b
  partition: <cluster-partition>
  time: 04:00:00
  cpus_per_gpu: 12
  slurm_resume: true
```

Shared container config:

```yaml
orchestrate:
  qwen-30b-a3b:
    gpus: 2
    tensor_parallel_size: 2
    serve:
      max_model_len: 40960
  vllm-container:
    image: vllm/vllm-openai:latest
    container_port: 8000
    volumes:
      - /data/huggingface:/root/.cache/huggingface
    ipc_mode: host
  pyxis:
    srun_extra_args: []
```

Config notes:

- `orchestrate.vllm-container` is the preferred key.
- `orchestrate.vllm-docker` is still accepted as a deprecated alias.
- Do not set both keys in the same job config.
- `orchestrate.<model>.gpus` is the minimum compatible outer allocation for the task, not the derived vLLM world size.
- `allocated_gpus` is the runtime allocation handed to the worker. Local Docker/Podman runs usually allocate exactly `gpus`; Slurm submission typically allocates `--node-gpus`.
- `data_parallel_size` is derived from `allocated_gpus // tensor_parallel_size`. If you set it explicitly, it must match that derived value or launch validation fails.
- `vllm_world_size` is `tensor_parallel_size * data_parallel_size`.
- Valid runtime launch shapes are `1`, `2`, `4`, and `8` GPUs.
- A launch is valid only when `allocated_gpus >= gpus`, `allocated_gpus >= tensor_parallel_size`, and `allocated_gpus % tensor_parallel_size == 0`.
- `ipc_mode` is Docker-only and is ignored in `--runtime pyxis`.
- `orchestrate.pyxis` is Pyxis-only and is ignored in `--runtime docker`.
- In Pyxis mode, Slurm allocates GPUs per `srun` step. The orchestrator only reserves localhost ports.
- Slurm-native submission only reads per-job `slurm:` defaults; plan-level `slurm:` defaults are not supported.

### CLI usage

```bash
medarc-orchestrate --plan plans/local-vllm.yaml
```

If you want queued Slurm submission rather than running inside your current shell or allocation, use:

```bash
medarc-orchestrate slurm --plan plans/local-vllm.yaml
```

Direct job config mode:

```bash
medarc-orchestrate --job-config configs/job-gpt-oss-20b.yaml
medarc-orchestrate \
  --job-config configs/job-gpt-oss-20b.yaml \
  --job-config configs/job-qwen-30b-a3b.yaml \
  --name local-vllm
```

Runtime examples:

```bash
medarc-orchestrate --plan plans/local-vllm.yaml --runtime docker
medarc-orchestrate --plan plans/local-vllm.yaml --runtime pyxis
```

Slurm-native submission examples:

```bash
medarc-orchestrate slurm --plan plans/local-vllm.yaml
medarc-orchestrate slurm \
  --job-config configs/job-gpt-oss-20b.yaml \
  --job-config configs/job-qwen-30b-a3b.yaml \
  --name local-vllm \
  --partition <cluster-partition> \
  --time 04:00:00
```

Common flags:

- `--runtime {docker,pyxis}` selects the serve backend. Default: CLI value, else `plan.runtime`, else `docker`.
- `--job-config PATH` runs directly from a job config. Repeat to launch multiple jobs without a wrapper plan file.
- `--name NAME` sets the bundle name when using `--job-config` directly.
- `--dry-run` prints resolved tasks and exits.
- `--gpu-range 0-3` restricts GPU indices (overrides `gpu_range` from the plan file).
- `--port-range 8000-8999` restricts ports (overrides `port_range` from the plan file).
- `--run-id` sets a custom run identifier (overrides `run_id` from the plan file). When omitted, `medarc-orchestrate` now uses the same default generator as `medarc-eval bench`: explicit `run_id`, else plan `run_id`, else `<slug(name)>-<timestamp>`, else `run-<timestamp>`.
- `--output-dir` overrides the output root (overrides `output_dir` from the plan file).
- `--max-parallel` caps concurrent tasks (overrides `max_parallel` from the plan file; defaults to GPU count).
- `--readiness-timeout-s` controls server readiness wait (overrides `readiness_timeout_s` from the plan file).
- `--resume` skips completed tasks using `summary.json` (enables resume even if `resume: false` in the plan).
- `--rerun-failed` reruns failed tasks on resume (enables rerun even if `rerun_failed: false` in the plan).
- `--status` prints the latest summary status and exits.
- `--kill-orphans` cleans up containers labeled as orchestrator-managed (also enabled by `kill_orphans: true` in the plan).
- `--prune-logs-on-success` deletes per-task `serve/container_logs.txt` and `bench/stdout.txt`+`stderr.txt` for completed tasks.

`medarc-orchestrate slurm` flags:

- `--node-gpus` sets the outer Slurm GPU allocation per submitted job. Default: `8`.
- `--max-simultaneous-nodes` limits the number of dependency chains. Default: `1`.
- `--run-simultaneously` removes generated inter-task dependencies.
- `--cpus-per-gpu`, `--time`, `--partition`, `--account`, `--qos`, `--mail-type`, and `--mail-user` override per-job `slurm:` defaults.
- When no account is provided, Slurm-native submission defaults to `training` and passes it on the `sbatch` command line.
- `--dependency` applies a base sbatch dependency to each chain head.
- `--slurm-resume` adds `#SBATCH --requeue` and passes `--resume` to the inner orchestrator.
- `--test-only` runs `sbatch --test-only` instead of submitting jobs.
- `--dry-run` writes scripts/configs and prints the `sbatch` commands without submitting them.
- `--source-dir` points at the checkout used inside the generated `sbatch` script. Default: current working directory.
- `--activate-script` overrides the shell activation script sourced by the generated `sbatch` script. Default: `<source-dir>/.venv/bin/activate`.

Slurm-native behavior:

- Each resolved orchestrator task becomes one generated `sbatch` script.
- Each generated `sbatch` script sources the configured activation script, then runs `medarc-orchestrate --no-uv-run ...`.
- Tasks are ordered by largest minimum `gpus` first, with larger `tensor_parallel_size` as the tie-breaker.
- Every job requests the full `--node-gpus` allocation with `#SBATCH --gpus-per-task=<node_gpus>`.
- The worker uses the full outer allocation it receives; it does not shrink a larger Slurm allocation down to the task minimum.
- Slurm submission keeps the bundled task config’s authored `gpus` value unchanged and records the runtime `allocated_gpus` separately in the execution allocation / manifests.
- The shared topology derivation computes `data_parallel_size = allocated_gpus // tensor_parallel_size` and `vllm_world_size = tensor_parallel_size * data_parallel_size`.
- Submission and launch validation fail immediately when `allocated_gpus` is not one of `1`, `2`, `4`, `8`, when it is smaller than the task’s minimum `gpus`, or when it is incompatible with `tensor_parallel_size`.
- When `orchestrate.restart` is absent, the task-local config persists an auto-generated restart target equal to the inner bench run id. That value is later passed to `medarc-eval bench --restart ...`, which resolves it under the default raw output root (`runs/raw/<bench-run-id>`).
- The task-local config is the durable source of truth for restart settings and shields submitted jobs from later edits to the original repo config.

Plan files may also set:

```yaml
runtime: pyxis
```

### Outputs

Artifacts are written under `outputs/orchestrator/<run_id>/`:

- `summary.json` aggregates task states.
- per-task folders contain `run_manifest.json`, `serve/` logs, `bench/` outputs, and `result.json`.

Slurm submission bundles are written under `outputs/slurm/<run_id>/` by default:

- `manifest.json` records the generated scripts, patched configs, dependency expressions, restart source, and submitted Slurm job IDs.
- `<task-slug>/slurm/orchestrate.sh` is the generated `sbatch` script for one orchestrator task.
- `<task-slug>/slurm/job-config.yaml` is always written as the task-local config used by the generated script.
- `<task-slug>/orchestrator/` contains the inner orchestrator artifacts, including the per-task `run_manifest.json` with GPU-hour accounting.

### Runtime behavior

Docker mode:

- The orchestrator reserves concrete local GPU IDs and host ports.
- vLLM listens on `container_port` inside the container and is mapped to a reserved localhost port.

Pyxis mode:

- The orchestrator assumes it is already running on the target GPU node within a Slurm allocation.
- Each task launches vLLM with `srun --container-image ... vllm serve --host 127.0.0.1 --port <reserved-port> ...`.
- Base URLs stay on localhost, for example `http://127.0.0.1:8123/v1`.
- `max_parallel > 1` is supported by reserving different localhost ports per task.
