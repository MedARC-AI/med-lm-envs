# Single-Run Mode

Run a single benchmark against one model. This is the fastest way to test a model or explore a benchmark.

This mode wraps the verifiers [`vf-eval`](https://github.com/primeintellect-ai/verifiers) command, adding automatic discovery of environment-specific flags from each benchmark's `load_environment()` function.

## Quick Start

```bash
# Basic: run 25 MedQA questions with GPT-4.1-mini
medarc-eval medqa -m gpt-4.1-mini -n 25

# See all options for an environment
medarc-eval medqa --help

# Preview without running
medarc-eval medqa -m gpt-4.1-mini -n 25 --dry-run
```

> **Remember:** The environment name must come first. `medarc-eval --verbose medqa` won't work.

## Common Options

### Model Selection

| Flag | Description |
|------|-------------|
| `-m`, `--model` | Model identifier (e.g., `gpt-4.1-mini`, `openai/gpt-4o`) |
| `-b`, `--api-base-url` | API endpoint URL (default: OpenAI) |
| `-k`, `--api-key-var` | Environment variable containing API key (default: `OPENAI_API_KEY`) |

### Run Configuration

| Flag | Description |
|------|-------------|
| `-n`, `--num-examples` | Number of examples to evaluate (default: 5, use -1 for all) |
| `-r`, `--rollouts-per-example` | Samples per example (default: 1) |
| `-c`, `--max-concurrent` | Max parallel API requests |
| `--timeout` | Request timeout in seconds |

### Sampling Parameters

| Flag | Description |
|------|-------------|
| `--temperature` | Sampling temperature |
| `--top-p` | Top-p (nucleus) sampling |
| `--top-k` | Top-k sampling |
| `--max-tokens` | Maximum tokens to generate |

### Output Control

| Flag | Description |
|------|-------------|
| `--save-results` / `--no-save-results` | Save outputs (default: enabled) |
| `--save-every N` | Save checkpoint every N rollouts |
| `-v`, `--verbose` | Enable debug logging |

## Environment-Specific Options

Each benchmark has its own options based on its `load_environment()` function. These appear automatically when you run `--help`:

```bash
medarc-eval longhealth --help
```

Example output:
```
longhealth options:
  --task {task1,task2,all}     Which task(s) to load (default: task1)
  --max-context-tokens INT     Maximum tokens for document context (default: 16000)
  --shuffle-docs / --no-shuffle-docs
                               Whether to shuffle document order (default: enabled)
  --shuffle-answers / --no-shuffle-answers
                               Whether to shuffle multiple-choice answers (default: disabled)
```

### Passing Complex Arguments

For arguments that can't be expressed as simple flags (dicts, nested structures), use JSON:

```bash
# Single JSON object
medarc-eval careqa --env-args '{"split": "open", "judge_model": "gpt-4o"}'

# Key=value pairs (repeatable)
medarc-eval careqa --env-arg split=open --env-arg judge_model=gpt-4o
```

**Precedence** (highest wins):
1. Explicit flags (`--shuffle-seed 42`)
2. `--env-arg key=value` pairs
3. `--env-args '{...}'` JSON

## Examples

### Run with a Local Model

```bash
medarc-eval medqa \
  -m openai/my-local-model \
  -b http://localhost:8000/v1 \
  -n 100
```

### Evaluate with Answer Shuffling

Many benchmarks support answer shuffling to test robustness:

```bash
medarc-eval medqa \
  -m gpt-4.1-mini \
  -n 100 \
  --shuffle-answers \
  --shuffle-seed 1618
```

### Use a Judge Model for Open-Ended Tasks

```bash
medarc-eval careqa \
  -m gpt-4.1-mini \
  -n 50 \
  --env-args '{"split": "open", "judge_model": "gpt-4o-mini"}'
```

### High-Throughput Evaluation

```bash
medarc-eval pubmedqa \
  -m gpt-4.1-mini \
  -n -1 \                    # All examples
  --max-concurrent 20 \      # 20 parallel requests
  --save-every 100           # Checkpoint every 100 samples
```

## Available Benchmarks

Run any environment from the `environments/` directory:

| Environment | Description |
|-------------|-------------|
| `medqa` | USMLE-style medical QA |
| `pubmedqa` | Biomedical literature QA |
| `medcalc_bench` | Medical calculation problems |
| `medxpertqa` | Expert-level medical QA |
| `longhealth` | Long-context medical reasoning |
| `healthbench` | OpenAI's health benchmark |
| `careqa` | CareQA (MCQ and open-ended) |
| `medagentbench` | Multi-turn agent benchmark (requires FHIR server) |

Use `medarc-eval <env> --help` to see environment-specific options.

If an environment isn't found, install it first with `vf-install <env>` (local) or `prime env install owner/env` (from Hub). See [Installing Environments](medarc-eval.md#installing-environments).
