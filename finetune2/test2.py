import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ============================================================
# 1. PATHS — UPDATE THIS ONLY
# ============================================================
MODEL_DIR = r"tinyllama-geocode-lora"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEST_FILE = "geocode_train_randomized2.jsonl"   # file containing 10 test prompts


# ============================================================
# 2. LOAD TOKENIZER (LOCAL ONLY, NO INTERNET)
# ============================================================
print("🧠 Loading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        legacy=False,
        local_files_only=True
    )
    print("✅ Loaded tokenizer from LoRA directory.")
except Exception as e:
    print("❌ Failed to load tokenizer from LoRA directory.\n", e)
    print("⚠ Falling back to base model tokenizer (NOT recommended!)")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        legacy=False
    )

# Ensure special token exists
if "<END>" not in tokenizer.get_vocab():
    print("⚠ WARNING: <END> token missing from tokenizer! Adding it now.")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})

END_ID = tokenizer.convert_tokens_to_ids("<END>")

# ============================================================
# 3. LOAD BASE MODEL + APPLY LORA ADAPTER
# ============================================================
print("\n🧠 Loading model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Check token count match
if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
    print(f"🔧 Resizing base model embeddings: {base_model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
    base_model.resize_token_embeddings(len(tokenizer))

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    torch_dtype=torch.float16,
    local_files_only=True
)

print("✅ LoRA adapter successfully loaded.")
model.eval()


# ============================================================
# 4. GENERATION FUNCTION
# ============================================================
def run_model(instruction, inp):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
            eos_token_id=END_ID,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Only extract content AFTER "### Response:"
    if "### Response:" in decoded:
        decoded = decoded.split("### Response:")[1].strip()

    # Stop at <END>
    if "<END>" in decoded:
        decoded = decoded.split("<END>")[0].strip()

    return decoded


# ============================================================
# 5. LOAD TEST INPUTS
# ============================================================
print("\n📄 Loading test input file...")

tests = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        tests.append(json.loads(line))

print(f"✅ Loaded {len(tests)} test items.\n")


# ============================================================
# 6. RUN TESTS
# ============================================================
for i, ex in enumerate(tests, 1):
    print(f"\n============================")
    print(f"🔍 TEST {i}")
    print("============================")
    instruction = ex["instruction"]
    inp = ex["input"]

    output = run_model(instruction, inp)
    print(output)

    # Extract inside_ids if they appear
    if "inside_ids:" in output:
        try:
            inside_part = output.split("inside_ids:")[1].strip()
            print(f"👉 Parsed inside_ids: {inside_part}")
        except:
            print("⚠ Could not parse inside_ids.")


print("\n🎉 Done running tests.")
