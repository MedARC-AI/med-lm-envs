# MedAgentBench V2

### Overview
- **Environment ID**: `medagentbenchv2`
- **Short description**: MedAgentBench V2 tool-calling environment (scaffold)
- **Tags**: medical, ehr, tool-calling, evaluation


### Prerequisites
Start the FHIR server:

```bash
docker pull jyxsu6/medagentbench:latest
docker tag jyxsu6/medagentbench:latest medagentbench
docker run --platform linux/amd64 \
  -e JAVA_TOOL_OPTIONS='-XX:+UseSerialGC -Xms256m -Xmx1024m' \
  -p 8080:8080 medagentbench:latest
```

### Smoke check
Run a tiny evaluation on 1–2 tasks:

```bash
uv run vf-eval medagentbenchv2 \
  -n 2 \
  -a '{"fhir_api_base":"http://localhost:8080/fhir/","task_types":["task1"]}'
```
