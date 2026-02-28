## vLLM orchestrator (Docker or Pyxis)

This repo includes an experimental vLLM orchestrator for running `medarc-eval` against
locally hosted vLLM servers with GPU/port scheduling.

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
- Run `medarc-orchestrate` inside an existing Slurm allocation on a GPU node.
- The Pyxis `--container-image` value must match cluster policy:
  - registry-backed URI such as `docker://vllm/vllm-openai:latest`
  - or a pre-staged squashfs image path

Notes for Pyxis:

- Recommended interactive workflow: `salloc --nodes=1 --gpus=4 --time=02:00:00`, then `uv run medarc-orchestrate --runtime pyxis ...`.
- `srun --pty bash` shells can block later `srun` steps on some clusters unless overlap is enabled. Prefer `salloc` or `sbatch`.
- Pulling large registry images at launch can fail on systems with limited local storage. Pre-staging a squashfs image is often more reliable.
- In shell scripts, image strings containing `#` may need escaping, for example `nvcr.io\\#nvidia/pytorch:25.01-py3`.

Note: This machine does not have Docker installed, so integration runs are not available here.

### Plan file

Create a plan YAML listing the job configs you want to orchestrate:

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
- `ipc_mode` is Docker-only and is ignored in `--runtime pyxis`.
- `orchestrate.pyxis` is Pyxis-only and is ignored in `--runtime docker`.
- In Pyxis mode, Slurm allocates GPUs per `srun` step. The orchestrator only reserves localhost ports.

### CLI usage

```bash
uv run medarc-orchestrate --plan plans/local-vllm.yaml
```

Runtime examples:

```bash
uv run medarc-orchestrate --plan plans/local-vllm.yaml --runtime docker
uv run medarc-orchestrate --plan plans/local-vllm.yaml --runtime pyxis
```

Common flags:

- `--runtime {docker,pyxis}` selects the serve backend. Default: CLI value, else `plan.runtime`, else `docker`.
- `--dry-run` prints resolved tasks and exits.
- `--gpu-range 0-3` restricts GPU indices (overrides `gpu_range` from the plan file).
- `--port-range 8000-8999` restricts ports (overrides `port_range` from the plan file).
- `--run-id` sets a custom run identifier (overrides `run_id` from the plan file).
- `--output-dir` overrides the output root (overrides `output_dir` from the plan file).
- `--max-parallel` caps concurrent tasks (overrides `max_parallel` from the plan file; defaults to GPU count).
- `--readiness-timeout-s` controls server readiness wait (overrides `readiness_timeout_s` from the plan file).
- `--resume` skips completed tasks using `summary.json` (enables resume even if `resume: false` in the plan).
- `--rerun-failed` reruns failed tasks on resume (enables rerun even if `rerun_failed: false` in the plan).
- `--status` prints the latest summary status and exits.
- `--kill-orphans` cleans up containers labeled as orchestrator-managed (also enabled by `kill_orphans: true` in the plan).
- `--prune-logs-on-success` deletes per-task `serve/container_logs.txt` and `bench/stdout.txt`+`stderr.txt` for completed tasks.

Plan files may also set:

```yaml
runtime: pyxis
```

### Outputs

Artifacts are written under `outputs/orchestrator/<run_id>/`:

- `summary.json` aggregates task states.
- per-task folders contain `run_manifest.json`, `serve/` logs, `bench/` outputs, and `result.json`.

### Runtime behavior

Docker mode:

- The orchestrator reserves concrete local GPU IDs and host ports.
- vLLM listens on `container_port` inside the container and is mapped to a reserved localhost port.

Pyxis mode:

- The orchestrator assumes it is already running on the target GPU node within a Slurm allocation.
- Each task launches vLLM with `srun --container-image ... vllm serve --host 127.0.0.1 --port <reserved-port> ...`.
- Base URLs stay on localhost, for example `http://127.0.0.1:8123/v1`.
- `max_parallel > 1` is supported by reserving different localhost ports per task.
