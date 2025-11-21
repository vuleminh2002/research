import json
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR   = "tinyllama-geocode-lora_s2"
TEST_FILE  = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 512
BATCH_SIZE = 1   # Increase later if you want huge TPS


print("=" * 70)
print("🚀 FAST INFERENCE — TinyLlama + LoRA (4-bit, RTX 4090)")
print("=" * 70)

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n📚 Loading tokenizer...")
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
    device_map={"": 0},   # force entire model to GPU
)

print("🔧 Applying LoRA adapters...")
model = PeftModel.from_pretrained(base, LORA_DIR)
model.eval()

print("✅ Model ready!")


# ============================================================
# GENERATION FUNCTION (TPS MEASURED)
# ============================================================

@torch.inference_mode()
def generate_response(prompt: str):
    """
    Run inference:
    - tokenizes prompt
    - generates model output
    - slices answer (skips prompt)
    - prints TPS
    """
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(model.device)

    prompt_len = encoded["input_ids"].shape[1]

    # Enable FlashAttention2 (if compiled in GPU stack)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
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

    # Extract only the newly generated tokens
    generated_ids = output[0][prompt_len:]
    gen_len = len(generated_ids)
    elapsed = end - start

    # Decode (fast path)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Token/s
    tps = gen_len / elapsed if elapsed > 0 else 0.0

    return text, gen_len, elapsed, tps


# ============================================================
# LOAD TEST DATA
# ============================================================

print(f"\n📂 Loading test data: {TEST_FILE}")
examples = [json.loads(l) for l in open(TEST_FILE, "r", encoding="utf-8")]
print(f"📊 Loaded {len(examples)} test prompts.")


# ============================================================
# RUN INFERENCE
# ============================================================

for idx, ex in enumerate(examples, start=1):
    print("\n" + "=" * 70)
    print(f"📝 Example {idx}/{len(examples)}")
    print("=" * 70)

    # Build full prompt for your model
    prompt = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )

    pred, gen_tokens, elapsed, tps = generate_response(prompt)

    print("\n🤖 MODEL OUTPUT")
    print("-" * 60)
    print(pred)

    print("\n🔢 STATS")
    print("-" * 60)
    print(f"Generated tokens : {gen_tokens}")
    print(f"Elapsed time     : {elapsed:.4f} sec")
    print(f"TPS (tok/sec)    : {tps:.2f}")

print("\n" + "=" * 70)
print("🎉 DONE — Fast inference complete.")
print("=" * 70)
