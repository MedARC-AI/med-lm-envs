# Running Medmarks on AWS Bedrock

Medmarks includes native support for AWS Bedrock via the Converse API. No proxy or external server required — just your AWS credentials.

## Prerequisites

- AWS account with Bedrock model access enabled in your target region
- AWS credentials configured (`aws configure`, SSO, or IAM role)
- Install the bedrock dependency group:

```bash
uv sync --group bedrock
```

## Quick Start

```bash
# Run MedQA with Nova Pro
uv run medarc-eval medqa \
  -m nova-pro \
  --endpoints-path configs/medmarks-endpoints-bedrock.toml \
  --provider bedrock \
  -n 25

# Run with a specific AWS profile
AWS_PROFILE=my-profile uv run medarc-eval medqa \
  -m nova-pro \
  --endpoints-path configs/medmarks-endpoints-bedrock.toml \
  --provider bedrock \
  -n 25

# Use a different region
uv run medarc-eval medqa \
  -m nova-pro \
  --endpoints-path configs/medmarks-endpoints-bedrock.toml \
  --provider bedrock \
  --api-base-url region:us-west-2 \
  -n 25
```

## Running Benchmark Suites

```bash
# Smoke test
uv run medarc-eval bench \
  --config configs/medmarks-smoke.toml \
  --endpoints-path configs/medmarks-endpoints-bedrock.toml \
  -m nova-pro \
  --provider bedrock \
  --dry-run

# Full verified suite
uv run medarc-eval bench \
  --config configs/medmarks-verified.toml \
  --endpoints-path configs/medmarks-endpoints-bedrock.toml \
  -m nova-pro \
  --provider bedrock
```

## Available Endpoint Aliases

| Alias | Bedrock Model ID |
|-------|-----------------|
| `nova-pro` | `us.amazon.nova-pro-v1:0` |
| `nova-lite` | `us.amazon.nova-lite-v1:0` |
| `nova-micro` | `us.amazon.nova-micro-v1:0` |
| `claude-sonnet-4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| `claude-haiku-3-5` | `us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| `claude-sonnet-3-5-v2` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |

## How It Works

Medmarks registers a `bedrock_converse` client type that calls the Bedrock [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) directly via boto3. Authentication uses the standard [boto3 credential chain](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html):

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. Shared credential file (`~/.aws/credentials`)
3. AWS SSO / IAM Identity Center (`aws sso login`)
4. EC2 instance role / ECS task role

No API keys are needed — SigV4 signing is handled automatically.

## Using Any Bedrock Model

Pass the Bedrock model ID directly (without an endpoint alias):

```bash
uv run medarc-eval medqa \
  -m us.meta.llama3-3-70b-instruct-v1:0 \
  --provider bedrock \
  -n 25
```

## Troubleshooting

**"AccessDeniedException"** — Enable model access in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) for your region.

**"ExpiredTokenException"** — Refresh your SSO session: `aws sso login --profile your-profile`

**Rate limiting on large benchmarks** — Reduce concurrency:
```bash
--max-concurrent 4
```
