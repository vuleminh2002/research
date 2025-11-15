import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# FORENSIC ANALYSIS OF TRAINED MODEL
# ============================================================

MODEL_DIR = "tinyllama-geocode-lora_final"  # Your already-trained model
DATA_FILE = "geocode_train_vary.jsonl"

print("=" * 60)
print("🔬 FORENSIC ANALYSIS: What Did The Model Learn?")
print("=" * 60)

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================
print("\n📂 Loading trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

print(f"✅ Model loaded from: {MODEL_DIR}")

# ============================================================
# LOAD TRAINING DATA
# ============================================================
print(f"\n📂 Loading training data: {DATA_FILE}")
with open(DATA_FILE) as f:
    train_data = [json.loads(line) for line in f]

sample_example = train_data[0]
print(f"✅ Loaded {len(train_data)} training examples")

# ============================================================
# TEST 1: What does the model generate with different prompts?
# ============================================================
print("\n" + "=" * 60)
print("🧪 TEST 1: Different Prompt Formats")
print("=" * 60)

test_input = sample_example["input"]
test_instruction = sample_example["instruction"]

prompts_to_test = [
    # Format 1: Standard format (what you used during training)
    {
        "name": "Standard Format",
        "prompt": f"### Instruction:\n{test_instruction}\n\n### Input:\n{test_input}\n\n### Response:\n"
    },
    
    # Format 2: With "Reasoning:" hint
    {
        "name": "With 'Reasoning:' Hint",
        "prompt": f"### Instruction:\n{test_instruction}\n\n### Input:\n{test_input}\n\n### Response:\nReasoning:\n"
    },
    
    # Format 3: Explicit instruction to show reasoning
    {
        "name": "Explicit Reasoning Request",
        "prompt": f"### Instruction:\n{test_instruction}\nYou MUST show your reasoning for each candidate.\n\n### Input:\n{test_input}\n\n### Response:\n"
    },
]

for test in prompts_to_test:
    print(f"\n{'─' * 60}")
    print(f"📝 {test['name']}")
    print(f"{'─' * 60}")
    
    inputs = tokenizer(test['prompt'], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"Generated (first 300 chars):")
    print(generated_text[:300])
    
    # Check what it contains
    has_reasoning = "Reasoning:" in generated_text
    has_arrow = "→" in generated_text
    has_inside_ids = "inside_ids:" in generated_text
    
    print(f"\nContent Analysis:")
    print(f"  Has 'Reasoning:': {has_reasoning}")
    print(f"  Has '→': {has_arrow}")
    print(f"  Has 'inside_ids:': {has_inside_ids}")

# ============================================================
# TEST 2: Check what tokens have high probability after prompt
# ============================================================
print("\n" + "=" * 60)
print("🧪 TEST 2: Token Probability Analysis")
print("=" * 60)

prompt = f"### Instruction:\n{test_instruction}\n\n### Input:\n{test_input}\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nAnalyzing what the model wants to generate first...")

with torch.no_grad():
    outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]  # Logits for next token
    
    # Get top 20 most likely next tokens
    top_k = 20
    top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), top_k)
    
    print(f"\nTop {top_k} most likely next tokens:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token = tokenizer.decode([idx])
        print(f"  {i:2d}. '{token}' (prob: {prob:.4f})")

# Check if "Reasoning" tokens are likely
reasoning_tokens = tokenizer("Reasoning:", add_special_tokens=False)["input_ids"]
inside_ids_tokens = tokenizer("inside_ids:", add_special_tokens=False)["input_ids"]

print(f"\nProbability of 'Reasoning:' tokens:")
for i, tok_id in enumerate(reasoning_tokens[:3]):  # First 3 tokens
    token_text = tokenizer.decode([tok_id])
    prob = torch.softmax(next_token_logits, dim=-1)[tok_id].item()
    print(f"  Token {i+1} ('{token_text}'): {prob:.6f}")

print(f"\nProbability of 'inside_ids:' tokens:")
for i, tok_id in enumerate(inside_ids_tokens[:3]):
    token_text = tokenizer.decode([tok_id])
    prob = torch.softmax(next_token_logits, dim=-1)[tok_id].item()
    print(f"  Token {i+1} ('{token_text}'): {prob:.6f}")

# ============================================================
# TEST 3: Reconstruct what training labels looked like
# ============================================================
print("\n" + "=" * 60)
print("🔬 TEST 3: Reconstruct Training Setup")
print("=" * 60)

RESPONSE_MARKER = "### Response:\n"

# Simulate your old tokenization function
def reconstruct_old_tokenization(ex):
    full_text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"{RESPONSE_MARKER}{ex['output']}"
    )
    
    tok = tokenizer(full_text, truncation=True, max_length=2048, padding=False)
    ids = tok["input_ids"]
    
    # Find marker
    marker_ids = tokenizer(RESPONSE_MARKER, add_special_tokens=False)["input_ids"]
    
    def find_subseq(main, sub):
        n, m = len(main), len(sub)
        for i in range(n - m + 1):
            if main[i:i+m] == sub:
                return i
        return -1
    
    marker_idx = find_subseq(ids, marker_ids)
    
    if marker_idx == -1:
        resp_start = max(0, len(ids) - 256)
    else:
        resp_start = marker_idx + len(marker_ids)
    
    # Build labels
    labels = []
    for i, t in enumerate(ids):
        if i < resp_start:
            labels.append(-100)
        else:
            labels.append(t)
    
    return {
        "input_ids": ids,
        "labels": labels,
        "resp_start": resp_start,
    }

print("\nReconstructing how training example was tokenized...")
reconstruction = reconstruct_old_tokenization(sample_example)

input_ids = reconstruction["input_ids"]
labels = reconstruction["labels"]
resp_start = reconstruction["resp_start"]

# Decode masked portion
masked_ids = [tok for i, tok in enumerate(input_ids) if labels[i] == -100]
learned_ids = [tok for i, tok in enumerate(input_ids) if labels[i] != -100]

masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
learned_text = tokenizer.decode(learned_ids, skip_special_tokens=False)

print(f"\nTotal tokens: {len(input_ids)}")
print(f"Masked tokens: {len(masked_ids)} ({len(masked_ids)/len(input_ids)*100:.1f}%)")
print(f"Learned tokens: {len(learned_ids)} ({len(learned_ids)/len(input_ids)*100:.1f}%)")
print(f"Response starts at token: {resp_start}")

print(f"\n📛 MASKED TEXT (what model DIDN'T learn from):")
print(f"{masked_text[:300]}")

print(f"\n✅ LEARNED TEXT (what model DID learn from):")
print(f"{learned_text[:500]}")

# CRITICAL CHECK
has_reasoning_in_learned = "Reasoning:" in learned_text
has_arrow_in_learned = "→" in learned_text

print(f"\n🔍 CRITICAL ANALYSIS:")
print(f"   'Reasoning:' in learned text: {has_reasoning_in_learned}")
print(f"   '→' in learned text: {has_arrow_in_learned}")

if not has_reasoning_in_learned:
    print(f"\n❌ SMOKING GUN FOUND!")
    print(f"   The training script MASKED OUT the reasoning!")
    print(f"   The model never saw 'Reasoning:' in the labels.")
    print(f"   This is why it doesn't generate reasoning.")
elif not has_arrow_in_learned:
    print(f"\n❌ PARTIAL PROBLEM FOUND!")
    print(f"   'Reasoning:' was learned, but arrows '→' were not.")
    print(f"   Check if arrows were in the training data.")
else:
    print(f"\n✅ Reasoning WAS in the learned portion.")
    print(f"   The problem might be elsewhere (data quality, overfitting, etc.)")

# ============================================================
# TEST 4: Check actual training data
# ============================================================
print("\n" + "=" * 60)
print("🔬 TEST 4: Verify Training Data Format")
print("=" * 60)

print("\nChecking first 3 training examples...")
for i in range(min(3, len(train_data))):
    ex = train_data[i]
    output = ex["output"]
    
    has_reasoning = "Reasoning:" in output
    has_arrow = "→" in output
    has_inside_ids = "inside_ids:" in output
    ends_with_eos = output.strip().endswith("</s>")
    
    print(f"\nExample {i+1}:")
    print(f"  Output length: {len(output)} chars")
    print(f"  Has 'Reasoning:': {has_reasoning}")
    print(f"  Has '→': {has_arrow}")
    print(f"  Has 'inside_ids:': {has_inside_ids}")
    print(f"  Ends with '</s>': {ends_with_eos}")
    print(f"  First 150 chars: {output[:150]}")
    
    if not has_reasoning or not has_arrow:
        print(f"  ⚠️  WARNING: Training data is malformed!")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("📊 FORENSIC SUMMARY")
print("=" * 60)

print("\n🔍 Key Findings:")
print(f"1. Model generates reasoning when prompted: {has_reasoning_in_learned}")
print(f"2. Training labels included 'Reasoning:': {has_reasoning_in_learned}")
print(f"3. Training labels included '→': {has_arrow_in_learned}")
print(f"4. Model prefers 'inside_ids:' over 'Reasoning:': [Check Test 2 output above]")

print("\n💡 Most Likely Root Cause:")
if not has_reasoning_in_learned:
    print("   ❌ MASKING ISSUE: Reasoning was masked out during training")
    print("      Solution: Retrain with reasoning in the learned portion")
elif has_reasoning_in_learned and "Reasoning:" not in generated_text[:50]:
    print("   ❌ SHORTCUT LEARNING: Model learned to skip reasoning")
    print("      The model found it could minimize loss by going straight to inside_ids")
    print("      Solution: Retrain with no masking OR add reasoning supervision")
else:
    print("   ⚠️  UNKNOWN: Need more investigation")

print("=" * 60)