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
LORA_DIR   = "/research/finetune2/tinyllama-geocode-lora_s2"  # local LoRA folder
TEST_FILE  = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 2048
BATCH_SIZE = 1

print("=" * 70)
print("🚀 FAST INFERENCE — TinyLlama + LoRA (4-bit)")
print("=" * 70)

# ============================================================
# GPU / DTYPE SETUP
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script.")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"\n🖥️  Using GPU: {gpu_name}")

# Use bf16 where supported (4090, H100), else fp16
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    print("🔢 Using bfloat16 for 4-bit compute.")
else:
    compute_dtype = torch.float16
    print("🔢 Using float16 for 4-bit compute.")

# H100-safe SDPA flag
USE_H100_SAFE_SDPA = "H100" in gpu_name

if USE_H100_SAFE_SDPA:
    print("⚠️ H100 detected — will use safe SDPA (no forced FlashAttention).")
else:
    print("✅ Non-H100 GPU — letting PyTorch choose best SDPA kernels.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

print(f"\n🔤 EOS token: {tokenizer.eos_token} (id={eos_id})")


# ============================================================
# LOAD BASE MODEL IN 4-BIT + APPLY LORA
# ============================================================

print("\n🧠 Loading base model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},  # single-GPU
)

print("🔧 Loading LoRA adapters from local folder...")
model = PeftModel.from_pretrained(
    base,
    LORA_DIR,
    is_trainable=False,
    local_files_only=True,
)
model.eval()

print("✅ Model ready!\n")


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

    # H100: force safe math SDPA to avoid weird kernel hangs
    if USE_H100_SAFE_SDPA:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

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
    tps = gen_len / elapsed if elapsed > 0 else 0.0

    return decoded, gen_len, elapsed, tps


# ============================================================
# ROBUST inside_ids EXTRACTION
# ============================================================

def extract_inside_ids(text: str):
    """
    Extract inside_ids from patterns like:

      inside_ids: ['a','b']
      inside_ids :
          ['a', 'b']
      inside_ids: []
      inside_ids: ['a','b']</s>

    Handles newlines, spaces, quotes, trailing </s>, etc.
    """
    text = text.replace("</s>", "").strip()
    match = re.search(r"inside_ids\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return []

    content = match.group(1).strip()
    if not content:
        return []

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

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 1.0

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

    pred_ids = extract_inside_ids(pred_text)
    gold_ids = extract_inside_ids(gold_text)

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
