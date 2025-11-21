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
LORA_DIR   = "/research/finetune2/tinyllama-geocode-lora_s2"
TEST_FILE  = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 1024   # Reduce for max speed
BATCH_SIZE = 1

torch.set_float32_matmul_precision("high")

print("=" * 70)
print("🚀 FASTEST POSSIBLE 4-BIT INFERENCE — H100 Optimized")
print("=" * 70)

# ============================================================
# GPU SETUP
# ============================================================

gpu_name = torch.cuda.get_device_name(0)
print(f"🖥️ GPU: {gpu_name}")

USE_H100 = "H100" in gpu_name
print(f"USE_H100 = {USE_H100}")

# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

# ============================================================
# LOAD BASE MODEL IN 4-BIT
# ============================================================

print("\n🧠 Loading base 4-bit model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
)

print("🔧 Loading LoRA adapters...")
model = PeftModel.from_pretrained(
    base,
    LORA_DIR,
    is_trainable=False,
    local_files_only=True,
)
model.eval()

print("✅ Model ready!\n")

# ============================================================
# H100-SAFE ATTENTION SETTINGS
# ============================================================

if USE_H100:
    print("⚠️ H100 detected — enabling safe (non-flash) SDPA")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

# ============================================================
# GENERATION (MAX SPEED FOR 4-BIT)
# ============================================================

@torch.inference_mode()
def generate_response(prompt: str):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # Warm KV cache
    _ = model(**inputs, use_cache=True)

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
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    tps = len(gen_ids) / (end - start)
    return text, len(gen_ids), (end - start), tps

# ============================================================
# inside_ids EXTRACTOR
# ============================================================

def extract_inside_ids(text):
    text = text.replace("</s>", "").strip()
    match = re.search(r"inside_ids\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return []
    items = match.group(1).strip()
    if not items:
        return []
    return [x.strip().strip("'").strip('"')
            for x in items.split(",") if x.strip()]

# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred_ids, gold_ids):
    p, g = set(pred_ids), set(gold_ids)
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 1.0
    return precision, recall, f1

# ============================================================
# LOAD TEST DATA
# ============================================================

examples = [json.loads(x) for x in open(TEST_FILE)]
print(f"📂 Loaded {len(examples)} examples\n")

# ============================================================
# RUN INFERENCE
# ============================================================

for idx, ex in enumerate(examples, 1):
    prompt = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )

    pred, gen_tokens, elapsed, tps = generate_response(prompt)

    gold_ids = extract_inside_ids(ex["output"])
    pred_ids = extract_inside_ids(pred)

    precision, recall, f1 = compute_metrics(pred_ids, gold_ids)

    print("\n" + "=" * 70)
    print(f"📝 Example {idx}")
    print("=" * 70)

    print("\n🤖 Output:\n", pred)
    print("\nGold:", ex["output"].strip())

    print("\nExtracted:")
    print("Pred:", pred_ids)
    print("Gold:", gold_ids)

    print("\nMetrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    print("\nSpeed:")
    print(f"Tokens: {gen_tokens}")
    print(f"Time:   {elapsed:.3f}s")
    print(f"TPS:    {tps:.2f}")

print("\n🎉 DONE!")
