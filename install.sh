#!/usr/bin/env bash
set -euo pipefail

command -v vf-install >/dev/null 2>&1 || {
  echo "error: vf-install not found in PATH" >&2
  exit 127
}

vf-install longhealth && \
  vf-install med_mcqa && \
  vf-install medbullets && \
  vf-install medconceptsqa && \
  vf-install medqa && \
  vf-install medxpertqa && \
  vf-install metamedqa && \
  vf-install mmlu_pro_health && \
  vf-install pubmedqa && \
  vf-install m_arc && \
  vf-install medcalc_bench && \
  vf-install careqa && \
  vf-install medagentbench && \
  vf-install head_qa_v2 && \
  vf-install med_halt && \
  vf-install supergpqa_medicine && \
  vf-install medhallu && \
  vf-install sctpublic && \
  vf-install pubhealthbench && \
  vf-install medec && \
  vf-install aci_bench && \
  vf-install medexqa && \
  vf-install med_dialog && \
  vf-install medrbench && \
  vf-install medcasereasoning && \
  vf-install medicationqa && \
  vf-install agentclinic && \
  vf-install mtsamples_procedures && \
  vf-install mtsamples_replicate && \
  vf-install healthbench && \
  vf-install medagentbenchv2 && \
  uv sync --inexact


echo "Done."
