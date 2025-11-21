import json
import time
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR   = "tinyllama-geocode-lora_s2"
TEST_FILE  = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 2048
BATCH_SIZE = 1

print("=" * 70)
print("🚀 FAST INFERENCE — TinyLlama + LoRA (4-bit, RTX 4090)")
print("=" * 70)

# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

# ============================================================
# LOAD BASE MODEL IN 4-BIT + APPLY LORA
# ============================================================

print("\n🧠 Loading base 4-bit model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
)

print("🔧 Loading LoRA adapters...")
model = PeftModel.from_pretrained(base, LORA_DIR)
model.eval()

print("✅ Model ready!")


# ============================================================
# GENERATION FUNCTION (TPS)
# ============================================================

@torch.inference_mode()
def generate_response(prompt: str):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    prompt_len = encoded["input_ids"].shape[1]

    # FlashAttention2 enabled
    with torch.backends.cuda.sdp_kernel(
        enable_math=False,
        enable_flash=True,
        enable_mem_efficient=True
    ):
        torch.cuda.synchronize()
        start = time.time()

        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
        )

        torch.cuda.synchronize()
        end = time.time()

    generated_ids = output[0][prompt_len:]
    gen_len = len(generated_ids)
    elapsed = end - start

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tps = gen_len / elapsed if elapsed > 0 else 0

    return decoded, gen_len, elapsed, tps


# ============================================================
# ROBUST inside_ids EXTRACTION (FIXED)
# ============================================================

def extract_inside_ids(text):
    """
    Extract inside_ids from ANY of the following patterns:
    
    inside_ids: ['a','b']
    inside_ids :
        ['a', 'b']
    inside_ids: []
    inside_ids: ['a','b']</s>

    Handles newlines, spaces, quotes, trailing </s>, etc.
    """
    # Remove special tokens
    text = text.replace("</s>", "").strip()

    # Regex: capture everything inside [ ... ]
    match = re.search(r"inside_ids\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return []

    content = match.group(1).strip()
    if not content:
        return []

    # Split by comma
    parts = content.split(",")

    ids = []
    for p in parts:
        token = p.strip().strip("'").strip('"')
        if token:
            ids.append(token)
    return ids


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred_ids, gold_ids):
    pred = set(pred_ids)
    gold = set(gold_ids)

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 1.0

    return precision, recall, f1


# ============================================================
# LOAD TEST DATA
# ============================================================

examples = [json.loads(line) for line in open(TEST_FILE, "r", encoding="utf-8")]
print(f"📂 Loaded {len(examples)} test examples\n")


# ============================================================
# RUN INFERENCE
# ============================================================

for idx, ex in enumerate(examples, start=1):

    prompt = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )

    pred_text, gen_tokens, elapsed, tps = generate_response(prompt)
    gold_text = ex["output"]

    # Extract IDs
    pred_ids = extract_inside_ids(pred_text)
    gold_ids = extract_inside_ids(gold_text)

    # Compute metrics
    precision, recall, f1 = compute_metrics(pred_ids, gold_ids)

    print("\n" + "=" * 70)
    print(f"📝 Example {idx}/{len(examples)}")
    print("=" * 70)

    print("\n🤖 MODEL OUTPUT:\n" + pred_text)
    print("\n🏷 TRUE OUTPUT:\n" + gold_text.strip())

    print("\n🔎 Extracted IDs:")
    print(f"Predicted: {pred_ids}")
    print(f"Gold     : {gold_ids}")

    print("\n📊 SCORES:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1       : {f1:.3f}")

    print("\n🚀 PERFORMANCE:")
    print(f"Tokens: {gen_tokens}")
    print(f"Time  : {elapsed:.3f}s")
    print(f"TPS   : {tps:.2f} tokens/s")

print("\n" + "=" * 70)
print("🎉 DONE — Fast inference + scoring complete")
print("=" * 70)