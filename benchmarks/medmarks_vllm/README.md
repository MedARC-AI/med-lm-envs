# MedMarks vLLM Request Benchmark

This directory generates faithful first-request inputs for non-agent MedMarks
vLLM benchmarking. The canonical dataset rows are OpenAI Chat Completions
request inputs, not flattened prompt strings.

The generator samples the current MedMarks vLLM subset:

- `longhealth`
- `careqa`
- `medcalc_bench` base/non-tool variants only
- `medhallu`
- `medxpertqa`
- `supergpqa_medicine`
- `medbullets`

Tool-enabled variants, including the `medcalc_bench` `tools` entry, are skipped
by default because this benchmark replays only the initial non-tool model
request.

## Artifacts

For each size, generation writes:

- `medmarks_vllm_<size>.requests.jsonl`: canonical OpenAI-native `messages`
  rows with provenance and no `prompt` or `output_tokens` fields.
- `medmarks_vllm_<size>.bench.toml`: run-level benchmark settings, including
  `max_tokens`.
- `medmarks_vllm_<size>.stats.json`: provenance plus prompt character and
  approximate input token summaries.

`prompt_chars` in stats is the sum of text content lengths across exported
messages. `input_tokens_approx` is a rough audit estimate over a deterministic
role/content rendering; it is not the target server's exact chat-template token
count unless you generate with an explicit tokenizer.

## Generate

```bash
UV_CACHE_DIR=.uv-cache \
uv run python benchmarks/medmarks_vllm/generate_dataset.py
```

To generate one explicit size/path:

```bash
UV_CACHE_DIR=.uv-cache \
uv run python benchmarks/medmarks_vllm/generate_dataset.py \
  --target-size 1000 \
  --output benchmarks/medmarks_vllm/medmarks_vllm_1k.requests.jsonl \
  --stats-output benchmarks/medmarks_vllm/medmarks_vllm_1k.stats.json \
  --bench-output benchmarks/medmarks_vllm/medmarks_vllm_1k.bench.toml \
  --max-tokens 512
```

Optional exact audit token counts:

```bash
UV_CACHE_DIR=.uv-cache \
uv run python benchmarks/medmarks_vllm/generate_dataset.py \
  --tokenizer /path/to/hf-tokenizer-or-model \
  --target-size 1000
```

If tokenizer loading would require a Hugging Face download, omit `--tokenizer`.

## Benchmark

Stock `vllm bench serve --dataset-name custom` is not the canonical path for
these inputs. In vLLM 0.21.0, custom JSONL rows are sent as one user prompt,
which flattens separate system/user messages and changes the request shape.

Use the MedMarks adapter instead:

```bash
UV_CACHE_DIR=.uv-cache \
uv run python benchmarks/medmarks_vllm/bench_client/run_requests.py \
  --bench-config benchmarks/medmarks_vllm/medmarks_vllm_500.bench.toml \
  --base-url http://127.0.0.1:8000 \
  --model <served-model-name> \
  --output outputs/medmarks_vllm_500.result.json
```

The adapter reads `*.requests.jsonl`, sends each row's `messages` unchanged to
`/v1/chat/completions`, applies the run-level `max_tokens`, and records latency,
throughput, failures, and any returned usage metadata. It refuses rows with
legacy `prompt` or `output_tokens` fields and rejects non-text content parts.

## Pyxis Client Image

Run the benchmark client from the materialized Pyxis image cache when measuring
cluster vLLM endpoints.

```bash
BASE_SQSH=/path/to/pyxis-images/vllm/latest.sqsh \
IMAGE_DIR=/path/to/pyxis-images/vllm-bench-client \
benchmarks/medmarks_vllm/bench_client/build_pyxis_image.sh
```

The script uses Enroot to create a writable sandbox from `BASE_SQSH`, installs
benchmark-only dependencies, and exports `IMAGE_DIR/latest.sqsh` for
`srun --container-image`.
