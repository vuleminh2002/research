import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "tinyllama-geocode-lora_s1"
MAX_NEW_TOKENS = 256  # small benchmark

print("="*70)
print("🔍 FLASH ATTENTION DIAGNOSTIC")
print("="*70)

# -----------------------------------------------------------
# Environment info
# -----------------------------------------------------------
print("\n📦 ENVIRONMENT INFO")
print("------------------------------------------------------------")
print(f"PyTorch version:   {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")
print(f"CUDA version:      {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU name:          {torch.cuda.get_device_name(0)}")
print("------------------------------------------------------------")

# -----------------------------------------------------------
# Load tokenizer
# -----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
prompt = "Hello, this is a benchmark test."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# -----------------------------------------------------------
# Helper benchmark function
# -----------------------------------------------------------
def benchmark(model):
    torch.cuda.synchronize()
    start = time.time()

    out = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    torch.cuda.synchronize()
    end = time.time()

    gen_tokens = out.shape[1] - input_ids.shape[1]
    tps = gen_tokens / (end - start)

    return gen_tokens, end - start, tps


# -----------------------------------------------------------
# Try loading model with FlashAttention2
# -----------------------------------------------------------
print("\n⚡ Checking FlashAttention2 availability...")
fa2_model = None

try:
    fa2_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    print("✅ FlashAttention2: ENABLED")
except Exception as e:
    print("❌ FlashAttention2 could NOT be enabled")
    print("Reason:")
    print(str(e))


# -----------------------------------------------------------
# Try loading model with normal attention
# -----------------------------------------------------------
print("\n🧠 Loading baseline model (default attention)...")

baseline_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("✅ Baseline model loaded")


# -----------------------------------------------------------
# Benchmark normal attention
# -----------------------------------------------------------
print("\n⏱ Benchmark: Normal Attention")
gen, t, tps = benchmark(baseline_model)
print("------------------------------------------------------------")
print(f"Generated tokens: {gen}")
print(f"Time:             {t:.3f} sec")
print(f"Tokens/sec:       {tps:.2f}")
print("------------------------------------------------------------")

# -----------------------------------------------------------
# Benchmark FlashAttention2 (if available)
# -----------------------------------------------------------
if fa2_model:
    print("\n⚡⏱ Benchmark: FlashAttention2")
    gen2, t2, tps2 = benchmark(fa2_model)
    print("------------------------------------------------------------")
    print(f"Generated tokens: {gen2}")
    print(f"Time:             {t2:.3f} sec")
    print(f"Tokens/sec:       {tps2:.2f}")
    print("------------------------------------------------------------")

    speedup = tps2 / tps if tps > 0 else 0
    print(f"\n🚀 SPEEDUP WITH FLASH ATTENTION: {speedup:.2f}×")
else:
    print("\n⚠ FlashAttention2 benchmark skipped (not available)")


print("\n🏁 DONE")
