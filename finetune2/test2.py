import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "tinyllama-geocode-lora_s1"
TEST_FILE = "geocode_train_vary_test.jsonl"
MAX_NEW_TOKENS = 1024

print("=" * 60)
print("🚀 H100 OPTIMIZED INFERENCE — TinyLlama Geocode")
print("=" * 60)


# ================================================================
# Load tokenizer normally
# ================================================================
print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
eos_id = tokenizer.eos_token_id
print(f"EOS token = {tokenizer.eos_token} (ID {eos_id})")


# ================================================================
# Load model using H100-optimized settings
# ================================================================
print("\n🧠 Loading model (BF16 + FlashAttention2 + BetterTransformer)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,      # H100-native format
    device_map="auto",
    low_cpu_mem_usage=True,          # avoids HeaderTooLarge
    trust_remote_code=True,
)

# FlashAttention2 (PyTorch 2.1+ / H100)
model = model.to_bettertransformer()

model.eval()
print("✅ Model loaded and optimized for H100\n")


# ================================================================
# Inference function
# ================================================================
def run_model(instruction, input_text):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = encoded["input_ids"].shape[1]

    with torch.no_grad(), torch.nn.attention.sdpa_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
        )

    generated = output[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=False)

    # Cut at EOS
    if "</s>" in text:
        text = text.split("</s>")[0]

    return text.strip(), len(generated)


# ================================================================
# Load and test all examples
# ================================================================
print(f"📂 Loading test set: {TEST_FILE}")
examples = [json.loads(line) for line in open(TEST_FILE)]
print(f"Test samples: {len(examples)}\n")

for idx, ex in enumerate(examples, start=1):
    print("\n" + "=" * 70)
    print(f"📝 TEST EXAMPLE {idx}/{len(examples)}")
    print("=" * 70)

    pred, gen_tokens = run_model(ex["instruction"], ex["input"])

    print("\n🤖 MODEL OUTPUT:")
    print("-" * 70)
    print(pred)

    print("\n🏷 TRUE OUTPUT:")
    print("-" * 70)
    print(ex["output"].replace("</s>", "").strip())

    print("\n🔢 TOKEN USAGE")
    print("-" * 70)
    print(f"Generated tokens:    {gen_tokens}")

print("\n" + "=" * 60)
print("🏁 DONE — All examples processed on H100")
print("=" * 60)
