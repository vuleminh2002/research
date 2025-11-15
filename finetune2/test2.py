import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "tinyllama-geocode-lora_s1"
TEST_FILE = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 1024
BATCH_SIZE = 1   # can increase to 2–8 for MAJOR speedup

print("=" * 70)
print("🚀 OPTIMIZED INFERENCE — TinyLlama Geocode")
print("=" * 70)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

print(f"EOS token = {tokenizer.eos_token} (ID {eos_id})")


# ============================================================
# LOAD MODEL — 4-bit QUANT (FASTEST)
# ============================================================

print("\n🧠 Loading model in 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("⚡ Compiling model with torch.compile()...")
model = torch.compile(model, mode="reduce-overhead")
model.eval()

print("✅ Model loaded & optimized!")


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_response(prompt: str):
    """Runs optimized inference, returns model text + token stats."""

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    prompt_len = encoded["input_ids"].shape[1]

    # enable FlashAttention2 if available
    with torch.backends.cuda.sdp_kernel(
        enable_math=False, enable_flash=True, enable_mem_efficient=True
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

    total_time = end - start
    generated_ids = output[0][prompt_len:]
    generated_len = len(generated_ids)

    # decode
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # find EOS
    eos_pos = text.find("</s>")
    if eos_pos != -1:
        text = text[:eos_pos].strip()

    tok_per_sec = generated_len / total_time if total_time > 0 else 0

    return text, generated_len, total_time, tok_per_sec


# ============================================================
# LOAD TEST FILE
# ============================================================

print(f"\n📂 Loading test file: {TEST_FILE}")
examples = [json.loads(line) for line in open(TEST_FILE, "r")]
print(f"📊 Total test examples: {len(examples)}")


# ============================================================
# RUN ALL EXAMPLES
# ============================================================

for idx, ex in enumerate(examples, start=1):
    print("\n" + "=" * 70)
    print(f"📝 TEST EXAMPLE {idx}/{len(examples)}")
    print("=" * 70)

    # Build prompt
    prompt = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )

    # Run model
    pred, gen_tokens, elapsed, tok_per_sec = generate_response(prompt)

    # Output results
    print("\n🟦 INPUT:")
    print("-" * 60)
    print(ex["input"])

    print("\n🤖 MODEL OUTPUT:")
    print("-" * 60)
    print(pred)

    print("\n🏷 TRUE OUTPUT:")
    print("-" * 60)
    print(ex["output"].replace("</s>", "").strip())

    print("\n🔢 TOKEN STATS")
    print("-" * 60)
    print(f"Generated tokens: {gen_tokens}")
    print(f"Time:            {elapsed:.3f} sec")
    print(f"Speed:           {tok_per_sec:.2f} tokens/sec")

print("\n" + "=" * 70)
print("✅ DONE — Optimized Inference Completed")
print("=" * 70)
