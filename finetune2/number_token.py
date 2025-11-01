from transformers import AutoTokenizer
import json
from tqdm import tqdm
import numpy as np

# 1️⃣ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2️⃣ Iterate through all examples
lengths = []
with open("geocode_train_randomized4.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Tokenizing examples"):
        ex = json.loads(line)
        text = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['output']}"
        )
        tokenized = tokenizer(text, truncation=False)
        lengths.append(len(tokenized["input_ids"]))

# 3️⃣ Print summary stats
print("\n--- Token Statistics ---")
print(f"Average tokens: {np.mean(lengths):.2f}")
print(f"Median tokens:  {np.median(lengths):.2f}")
print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
print(f"Max tokens: {max(lengths)}")
