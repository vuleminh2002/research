import json
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "tinyllama-geocode-lora_s1"     # merged model folder
TEST_FILE = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 1024
BATCH_SIZE = 1

print("=" * 70)
print("🚀 FAST INFERENCE — TinyLlama (Merged LoRA) 4-bit")
print("=" * 70)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id


# ============================================================
# LOAD MERGED MODEL IN 4-BIT
# ============================================================

print("\n🧠 Loading merged model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map={"": 0},
)

model.eval()
print("✅ Model ready!")


# ============================================================
# GENERATION FUNCTION
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

    gen_ids = output[0][prompt_len:]
    gen_len = len(gen_ids)
    elapsed = end - start
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    tps = gen_len / elapsed if elapsed > 0 else 0
    return text.strip(), gen_len, elapsed, tps


# ============================================================
# LOAD TEST DATA
# ============================================================

print(f"\n📂 Loading test file: {TEST_FILE}")
examples = [json.loads(line) for line in open(TEST_FILE, "r")]


# ============================================================
# RUN INFERENCE
# ============================================================

for idx, ex in enumerate(examples, start=1):
    print("\n" + "=" * 70)
    print(f"📝 TEST EXAMPLE {idx}/{len(examples)}")
    print("=" * 70)

    prompt = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )

    pred, gen_tokens, elapsed, tps = generate_response(prompt)

    print("\n🤖 MODEL OUTPUT:")
    print(pred)

    print("\n🔢 TOKEN STATS")
    print(f"Tokens: {gen_tokens}")
    print(f"Time:   {elapsed:.3f}s")
    print(f"TPS:    {tps:.2f}")

print("\n" + "=" * 70)
print("✅ DONE — Fast Inference Complete")
print("=" * 70)
