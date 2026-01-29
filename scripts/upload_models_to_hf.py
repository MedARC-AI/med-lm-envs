#!/usr/bin/env python3
"""Upload trained models to HuggingFace Hub."""

import os
from huggingface_hub import HfApi, upload_folder

# Models to upload
MODELS = {
    "nsk7153/Qwen3-4B-MedCombined-RL": {
        "path": "/admin/home/nikhil/med-lm-envs/prime-rl/outputs_combined/weights/step_500",
        "description": "Qwen3-4B fine-tuned with RL on combined medical datasets (MedCalc-Bench, MedMCQA, MedCaseReasoning)",
    },
    "nsk7153/Qwen3-4B-MedCalc-RL": {
        "path": "/admin/home/nikhil/med-lm-envs/prime-rl/outputs_verified_short/weights/step_300",
        "description": "Qwen3-4B fine-tuned with RL on MedCalc-Bench-Verified for medical calculations",
    },
    "nsk7153/Qwen3-4B-MedMCQA-RL": {
        "path": "/admin/home/nikhil/med-lm-envs/prime-rl/outputs_medmcqa/weights/step_300",
        "description": "Qwen3-4B fine-tuned with RL on MedMCQA for medical multiple choice QA",
    },
    "nsk7153/Qwen3-4B-MedCaseReasoning-RL": {
        "path": "/admin/home/nikhil/med-lm-envs/prime-rl/outputs_medcasereasoning/weights/step_300",
        "description": "Qwen3-4B fine-tuned with RL on MedCaseReasoning for clinical case analysis",
    },
}

# Model card template
MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
- medical
- reinforcement-learning
- qwen3
- healthcare
---

# {repo_name}

{description}

## Model Details

- **Base Model**: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Training Method**: Reinforcement Learning (GRPO)
- **Framework**: [verifiers](https://github.com/willieneis/verifiers) + [prime-rl](https://github.com/PRIME-RL/PRIME-RL)

## Training Data

{training_data}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Example usage
messages = [
    {{"role": "user", "content": "Your medical question here"}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## License

Apache 2.0
"""

TRAINING_DATA = {
    "nsk7153/Qwen3-4B-MedCombined-RL": """
This model was trained on a combination of three medical datasets:
- **MedCalc-Bench-Verified**: Medical calculation problems with verified numerical answers
- **MedMCQA**: Medical multiple choice questions from AIIMS/NEET PG exams
- **MedCaseReasoning**: Clinical case reasoning with LLM-as-judge evaluation
""",
    "nsk7153/Qwen3-4B-MedCalc-RL": """
This model was trained on **MedCalc-Bench-Verified**, a dataset of medical calculation problems 
requiring numerical reasoning about drug dosages, lab values, clinical scores, etc.
""",
    "nsk7153/Qwen3-4B-MedMCQA-RL": """
This model was trained on **MedMCQA**, a large-scale multiple choice question dataset 
covering various medical topics from AIIMS/NEET PG entrance exams.
""",
    "nsk7153/Qwen3-4B-MedCaseReasoning-RL": """
This model was trained on **MedCaseReasoning**, a dataset of clinical case analysis problems
evaluated using an LLM-as-judge approach for reasoning quality.
""",
}


def main():
    api = HfApi()
    
    for repo_id, info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Uploading: {repo_id}")
        print(f"From: {info['path']}")
        print(f"{'='*60}")
        
        # Create model card
        model_card = MODEL_CARD_TEMPLATE.format(
            repo_name=repo_id.split("/")[1],
            description=info["description"],
            training_data=TRAINING_DATA[repo_id],
            repo_id=repo_id,
        )
        
        # Write model card to the model directory
        readme_path = os.path.join(info["path"], "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        print(f"Created README.md")
        
        # Create the repo if it doesn't exist
        try:
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"Created/verified repo: {repo_id}")
        except Exception as e:
            print(f"Repo creation note: {e}")
        
        # Upload the folder
        print(f"Uploading files...")
        upload_folder(
            folder_path=info["path"],
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["STABLE"],  # Skip the STABLE marker file
        )
        print(f"✓ Successfully uploaded to https://huggingface.co/{repo_id}")
    
    print("\n" + "="*60)
    print("ALL UPLOADS COMPLETE!")
    print("="*60)
    for repo_id in MODELS:
        print(f"  https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
