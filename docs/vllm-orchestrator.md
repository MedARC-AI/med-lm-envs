## vLLM orchestrator (local Docker)

This repo includes an experimental vLLM orchestrator for running `medarc-eval` against
locally hosted vLLM Docker containers with GPU/port scheduling.

### Requirements

- Docker installed with NVIDIA runtime support.
- Local GPUs available (NVML via `nvidia-ml-py`).
- vLLM container images available locally or pullable.

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

### CLI usage

```bash
uv run medarc-orchestrate --plan plans/local-vllm.yaml
```

Common flags:

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

### Outputs

Artifacts are written under `outputs/orchestrator/<run_id>/`:

- `summary.json` aggregates task states.
- per-task folders contain `run_manifest.json`, `serve/` logs, `bench/` outputs, and `result.json`.
