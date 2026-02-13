# MedAgentBench V2

### Overview
- **Environment ID**: `medagentbenchv2`
- **Short description**: Tool-calling evaluation environment for clinical EHR tasks using FHIR APIs
- **Tags**: medical, ehr, tool-calling, multi-turn, clinical, evaluation

### Dataset
- **Primary dataset**: MedAgentBench V2 evaluation tasks
- **Source**: [MedAgentBench GitHub](https://github.com/stanfordmlgroup/MedAgentBench), [Paper](https://arxiv.org/abs/2501.14654)
- **Task variants**:
  - `new_patient_tasks`: Patient-centric clinical scenarios (default; matches upstream `collect_agent_responses.py`)
  - `new_tasks`: General clinical tasks
  - `test_data_v2`: Extended evaluation dataset
- **Task categories**: 10 task types covering various EHR operations (task1-task10)

### Task
- **Type**: multi-turn tool-calling
- **Parser**: Default parser (handles tool calls and finish tool)
- **Rubric overview**: Binary scoring (0 or 1) based on successful task completion using category-specific evaluation functions

### Note

This environment is a light verifiers wrapper around the original MedAgentBenchV2 code. The primary difference is the original code uses OpenAI's Responses API while this environment uses Chat Completions API through verifiers. The MedAgentBenchV2 prompts were lightly modified for generic tool calling versus the original's Responses API format examples.

### Prerequisites
Before running evaluations, you must start the FHIR server:

```bash
docker pull jyxsu6/medagentbench:latest
docker tag jyxsu6/medagentbench:latest medagentbench
docker run --platform linux/amd64 \
  -e JAVA_TOOL_OPTIONS='-XX:+UseSerialGC -Xms256m -Xmx1024m' \
  -p 8080:8080 medagentbench:latest
```

**Important**:
- The trailing slash in the FHIR URL is required
- Replace `localhost` with your actual IP address if running on a remote server
- Server connectivity is automatically verified before evaluation begins

### Quickstart
Run an evaluation with default settings (requires FHIR server):

```bash
vf-eval medagentbenchv2 \
  -a '{"fhir_api_base": "http://localhost:8080/fhir/"}'
```

Run a small evaluation on specific task types:

```bash
vf-eval medagentbenchv2 \
  -n 5 \
  -a '{"fhir_api_base": "http://localhost:8080/fhir/", "task_types": ["task1", "task2"]}'
```

Configure model and sampling:

```bash
vf-eval medagentbenchv2 \
  -m gpt-4.1-mini \
  -n 20 -r 1 -t 2048 -T 0 \
  -a '{"fhir_api_base": "http://localhost:8080/fhir/", "max_turns": 10}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object
- The FHIR server must be accessible at the specified URL
- Models should support tool calling for this environment

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `fhir_api_base` | str | Required | Base URL for the FHIR server (must include trailing slash) |
| `tasks_variant` | str | `"new_patient_tasks"` | Task variant to use: `"new_patient_tasks"`, `"new_tasks"`, or `"test_data_v2"` |
| `tasks_path` | str | None | Optional custom path to tasks JSON file (overrides `tasks_variant`) |
| `task_types` | list[str] | None | Optional list of task types to filter (e.g., `["task1", "task2"]`) |
| `max_turns` | int | 8 | Maximum number of interaction turns per task |

### Available Tools

The environment provides FHIR-based tools for clinical operations:
- `fhir_patient_search`: Search for patients in the EHR
- `fhir_observation_search`: Query clinical observations
- `fhir_vitals_search`: Search vital signs
- `fhir_vitals_create`: Create new vital sign records
- `fhir_medication_request_search`: Search medication requests
- `fhir_medication_request_create`: Create medication requests
- `fhir_procedure_search`: Query procedures
- `fhir_condition_search`: Search patient conditions
- `fhir_service_request_create`: Create service requests
- `finish`: Submit the final answer (required to complete tasks)

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `medagentbench_reward` | (weight 1.0): Binary score (1 if task correctly solved, 0 otherwise) |

### Task Categories

The environment evaluates 10 distinct task categories, each with specialized evaluation logic:
- **task1-task10**: Various EHR operations including patient search, data retrieval, record creation, and clinical decision support
