import json
import time
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "/research/finetune2/tinyllama-geocode-merged-s2"
TEST_FILE = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 2048   # you can tune this
BATCH_SIZE = 1

torch.set_float32_matmul_precision("high")

print("=" * 70)
print("🚀 H100-OPTIMIZED INFERENCE — Merged BF16 TinyLlama")
print("=" * 70)

# ============================================================
# GPU SETUP
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"🖥️ GPU: {gpu_name}")

IS_H100 = "H100" in gpu_name
print(f"IS_H100 = {IS_H100}")

# ============================================================
# TOKENIZER + MODEL
# ============================================================

print("\n🔤 Loading tokenizer & merged model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)

# Use PyTorch SDPA (flash/mem-efficient) where possible
try:
    # Some HF versions support this flag
    model.config.use_cache = True
except Exception:
    pass

model.eval()

print("✅ Model loaded.\n")

# Optional: try compile on H100 (comment out if it ever misbehaves)
try:
    print("⚙️ Compiling model with torch.compile()…")
    model = torch.compile(model, mode="max-autotune")
    print("✅ Compile enabled.")
except Exception as e:
    print(f"⚠️ torch.compile failed ({e}), continuing without compile.")

# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate_response(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Let PyTorch pick the best SDPA kernels (H100 will use TE/flash)
    torch.cuda.synchronize()
    start = time.time()

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        do_sample=False,
        use_cache=True,
    )

    torch.cuda.synchronize()
    end = time.time()

    gen_ids = output[0][prompt_len:]
    gen_len = len(gen_ids)
    elapsed = end - start

    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    tps = gen_len / elapsed if elapsed > 0 else 0.0

    return text, gen_len, elapsed, tps

# ============================================================
# inside_ids EXTRACTOR
# ============================================================

def extract_inside_ids(text: str):
    text = text.replace("</s>", "").strip()
    m = re.search(r"inside_ids\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        return []
    content = m.group(1).strip()
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
    p, g = set(pred_ids), set(gold_ids)
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)

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
print("🎉 DONE — H100-optimized BF16 inference complete")
print("=" * 70)
