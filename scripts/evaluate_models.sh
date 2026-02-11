#!/bin/bash
# Evaluate trained RL models on test sets using medarc-eval
#
# Usage:
#   ./evaluate_models.sh                    # Evaluate all models on all datasets
#   ./evaluate_models.sh --model medcalc    # Evaluate specific model
#   ./evaluate_models.sh --dataset medmcqa  # Evaluate on specific dataset
#
# Prerequisites:
#   1. Start vLLM server for the model you want to evaluate:
#      cd ~/med-lm-envs/prime-rl
#      CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve <model_path> --tensor-parallel-size 2 --port 8000
#
#   2. Run this script with the API pointing to the vLLM server

set -e

# Configuration
N_SAMPLES=100
MAX_CONCURRENT=10
TEMPERATURE=0.0
MAX_TOKENS=1024
API_BASE="http://localhost:8000/v1"

# Model paths
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
MEDCALC_MODEL="/admin/home/nikhil/med-lm-envs/prime-rl/outputs_verified_short/weights/step_300"
MEDMCQA_MODEL="/admin/home/nikhil/med-lm-envs/prime-rl/outputs_medmcqa/weights/step_300"
MEDCASE_MODEL="/admin/home/nikhil/med-lm-envs/prime-rl/outputs_medcasereasoning/weights/step_300"

# Output directory
OUTPUT_DIR="/admin/home/nikhil/med-lm-envs/eval_results"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Model Evaluation Script"
echo "=========================================="
echo "Samples per dataset: $N_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to run evaluation
run_eval() {
    local model_name=$1
    local dataset=$2
    local extra_args=$3
    
    echo ">>> Evaluating $model_name on $dataset"
    
    output_file="$OUTPUT_DIR/${model_name}_${dataset}.json"
    
    cd /admin/home/nikhil/med-lm-envs
    uv run medarc-eval "$dataset" \
        -m "openai/model" \
        -b "$API_BASE" \
        -n "$N_SAMPLES" \
        --max-concurrent "$MAX_CONCURRENT" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        $extra_args \
        2>&1 | tee "$OUTPUT_DIR/${model_name}_${dataset}.log"
    
    echo ">>> Finished $model_name on $dataset"
    echo ""
}

# Print instructions
cat << 'EOF'
========================================
INSTRUCTIONS
========================================

To evaluate a model, you need to:

1. Start a vLLM server for the model:

   # For base model:
   cd ~/med-lm-envs/prime-rl
   CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
       --tensor-parallel-size 2 --port 8000

   # For MedCalc-trained model:
   CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve \
       /admin/home/nikhil/med-lm-envs/prime-rl/outputs_verified_short/weights/step_300 \
       --tensor-parallel-size 2 --port 8000

   # For MedMCQA-trained model:
   CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve \
       /admin/home/nikhil/med-lm-envs/prime-rl/outputs_medmcqa/weights/step_300 \
       --tensor-parallel-size 2 --port 8000

   # For MedCaseReasoning-trained model:
   CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve \
       /admin/home/nikhil/med-lm-envs/prime-rl/outputs_medcasereasoning/weights/step_300 \
       --tensor-parallel-size 2 --port 8000

2. In another terminal, run the evaluation:

   # MedCalc-Bench-Verified
   cd ~/med-lm-envs
   uv run medarc-eval medcalc_bench \
       -m openai/model \
       -b http://localhost:8000/v1 \
       -n 100 \
       --temperature 0.0 \
       --use-think \
       --use-verified-dataset

   # MedMCQA
   uv run medarc-eval med_mcqa \
       -m openai/model \
       -b http://localhost:8000/v1 \
       -n 100 \
       --temperature 0.0 \
       --use-think

   # MedCaseReasoning (requires judge API key)
   export OPENAI_API_KEY=<your-key>
   uv run medarc-eval medcasereasoning \
       -m openai/model \
       -b http://localhost:8000/v1 \
       -n 100 \
       --temperature 0.0

========================================
EOF
