from datasets import load_dataset
import tiktoken
from statistics import mean

encoding = tiktoken.get_encoding("o200k_base")

INSTRUCTION = "Give a letter answer among A, B, C or D."

def build_prompt(ex):
    return (
        f"{INSTRUCTION} \n"
        f"Question: {ex['question'].strip()} \n"
        f"A. {ex['opa'].strip()} \n"
        f"B. {ex['opb'].strip()} \n"
        f"C. {ex['opc'].strip()} \n"
        f"D. {ex['opd'].strip()} \n" 
        f"Answer:"
    )
    
def count_tokens(text):
    return len(encoding.encode(text))

def token_stats(split):
    ds = load_dataset("lighteval/med_mcqa", split=split)
    counts = []
    for ex in ds:
        text = build_prompt(ex)
        counts.append(count_tokens(text))
    
    total_tokens = sum(counts)
    mean_tokens = mean(counts)
    print(f"{split} - examples: {len(counts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Mean tokens per example: {mean_tokens:.2f} \n")
    return counts
    

if __name__ == "__main__":
    print("MedMCQA token counts (o200k_base) \n")
    train_counts = token_stats("train")
    val_counts = token_stats("validation")
    combined = train_counts + val_counts
    print("Combined train+validation:")
    print(f"Total tokens: {sum(combined)}")
    print(f"Mean tokens per example: {mean(combined)}")