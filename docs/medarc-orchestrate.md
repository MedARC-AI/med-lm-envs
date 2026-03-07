## vLLM orchestrator (`medarc-orchestrate`)

`medarc-orchestrate` has two public execution modes:

- `medarc-orchestrate local`
- `medarc-orchestrate slurm`

The older top-level public forms were removed. Commands such as `medarc-orchestrate --plan ...` and `medarc-orchestrate --job-config ...` now fail in the parser; use `local` or `slurm` explicitly.

The internal worker entrypoint still exists for generated Slurm scripts:

```bash
medarc-orchestrate worker --task ... --allocation ... --runtime pyxis
```

It is not part of the public CLI.

### Execution modes

`local` runs the worker directly from your current shell state.

- Default runtime selection is automatic: prefer `docker`, fall back to `podman`.
- `local --runtime pyxis` is still supported when you are already inside a Slurm allocation and want the worker to launch with `srun --container-image ...`.

`slurm` is the queued submission path.

- It writes one `sbatch` script per resolved task.
- Generated jobs invoke `medarc-orchestrate worker`.
- The runtime on the compute node defaults to `pyxis`.

### Requirements

All modes:

- `medarc-orchestrate` runs `medarc-eval bench` as a local subprocess on the same host.
- The configured container image must be available to the selected runtime.

`local` with Docker:

- `docker` available on `PATH`
- NVIDIA runtime support
- local GPUs available through NVML / `nvidia-ml-py`
- Python `docker` package installed

`local` with Podman:

- `podman` available on `PATH`
- local GPUs available through NVML / `nvidia-ml-py`

`local --runtime pyxis` or `slurm`:

- Slurm with Pyxis installed
- Enroot available on compute nodes
- a valid Pyxis `--container-image` value for the cluster

### Inputs

Both public commands accept either a reusable plan file or one or more direct job configs:

```bash
medarc-orchestrate local --plan plans/local-vllm.yaml
medarc-orchestrate local --job-config configs/job-gpt-oss-20b.yaml

medarc-orchestrate slurm --plan plans/local-vllm.yaml
medarc-orchestrate slurm --job-config configs/job-gpt-oss-20b.yaml
```

Direct job-config mode can be repeated:

```bash
medarc-orchestrate local \
  --job-config configs/job-a.yaml \
  --job-config configs/job-b.yaml \
  --name local-vllm

medarc-orchestrate slurm \
  --job-config configs/job-a.yaml \
  --job-config configs/job-b.yaml \
  --name cluster-batch
```

Each job config must define exactly one model under `models:` and include a top-level `orchestrate:` block with per-model serve settings.

Example:

```yaml
models:
  qwen-30b-a3b:
    model: Qwen/Qwen3-30B-A3B

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

slurm:
  partition: gpu
  time: 04:00:00
  cpus_per_gpu: 12
  slurm_resume: true
```

Config notes:

- `orchestrate.vllm-container` is the preferred key.
- `orchestrate.vllm-docker` is still accepted as a deprecated config alias.
- `orchestrate.<model>.gpus` is the minimum compatible outer allocation.
- `allocated_gpus` is the runtime allocation actually handed to the worker.
- `data_parallel_size` is derived as `allocated_gpus // tensor_parallel_size`.
- `vllm_world_size` is `tensor_parallel_size * data_parallel_size`.
- Valid launch shapes are `1`, `2`, `4`, and `8` GPUs.
- Launch validation fails when `allocated_gpus < gpus`, `allocated_gpus < tensor_parallel_size`, or `allocated_gpus % tensor_parallel_size != 0`.

### `local`

Typical examples:

```bash
medarc-orchestrate local --plan plans/local-vllm.yaml
medarc-orchestrate local --plan plans/local-vllm.yaml --runtime podman
medarc-orchestrate local --plan plans/local-vllm.yaml --runtime pyxis
```

Default runtime behavior:

- If `--runtime` is set, that wins.
- Otherwise `plan.runtime` is used when present.
- Otherwise `local` prefers `docker` when it is available and falls back to `podman`.

Important `local` flags:

- `--plan PATH` or repeated `--job-config PATH`
- `--name NAME` for direct job-config mode
- `--runtime {docker,podman,pyxis}`
- `--env-file PATH`
- `--dry-run`
- `--gpu-range 0-3`
- `--port-range 8000-8999`
- `--run-id RUN_ID`
- `--output-dir PATH`
- `--max-parallel N`
- `--readiness-timeout-s SECONDS`
- `--resume`
- `--rerun-failed`
- `--status`
- `--kill-orphans`
- `--prune-logs-on-success`
- `--no-uv-run`

### `slurm`

Typical examples:

```bash
medarc-orchestrate slurm --plan plans/local-vllm.yaml

medarc-orchestrate slurm \
  --job-config configs/job-a.yaml \
  --job-config configs/job-b.yaml \
  --partition gpu \
  --time 04:00:00
```

Important `slurm` flags:

- `--plan PATH` or repeated `--job-config PATH`
- `--name NAME` for direct job-config mode
- `--run-id RUN_ID`
- `--output-dir PATH`
- `--env-file PATH`
- `--readiness-timeout-s SECONDS`
- `--prune-logs-on-success`
- `--node-gpus N` to set the outer Slurm allocation per task job
- `--max-simultaneous-nodes N`
- `--run-simultaneously`
- `--cpus-per-gpu`, `--time`, `--partition`, `--account`, `--qos`, `--mail-type`, `--mail-user`
- `--dependency` for chain heads
- `--test-only`
- `--dry-run`
- `--slurm-resume`
- `--source-dir PATH`
- `--activate-script PATH`

`slurm` behavior:

- Each resolved task becomes one generated `sbatch` script.
- Generated scripts source the activation script, then run `medarc-orchestrate worker`.
- Execution on the compute node uses `--runtime pyxis`.
- Tasks are ordered by largest minimum `gpus` first, with `tensor_parallel_size` as the tie-breaker.

### Outputs

Orchestrator-owned artifacts are written under `outputs/orchestrate/<run_id>/`.

Typical files:

- `run_manifest.json`
- `summary.json`
- `tasks/<task-slug>/task.yaml`
- `tasks/<task-slug>/eval-config.yaml`
- `tasks/<task-slug>/runtime/allocation.json`
- `tasks/<task-slug>/runtime/state.json`
- `tasks/<task-slug>/submit.sh` for Slurm submission

`medarc-eval bench` raw outputs still live under `runs/raw/<bench_run_id>/`.
