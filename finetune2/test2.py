import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import os

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "tinyllama-geocode-lora_v3/checkpoint-339"
TEST_FILE = "geocode_train_vary_test.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# LOAD TOKENIZER
# ======================================================
print("🧠 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if "<END>" not in tokenizer.get_vocab():
    print("⚠ Adding <END> token")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})

tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# LOAD MODEL + LORA
# ======================================================
print("🧠 Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.to(DEVICE)
model.eval()

# ======================================================
# GENERATE
# ======================================================
def generate_response(instr, inp, debug_id):
    prompt = (
        f"### Instruction:\n{instr}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )

    # ---- DEBUG: Show prompt + token count ----
    print("\n🔍 Full prompt provided to the model:")
    print(prompt)

    tokens = tokenizer(prompt, return_tensors="pt")
    token_count = len(tokens["input_ids"][0])
    print(f"🔢 Token count = {token_count}")

    if token_count > 2048:
        print("❌ ERROR: Prompt exceeds TinyLlama's context limit (2048 tokens)")
        print("   → This WILL cause hallucinations and garbage output.")
    
    # Save prompt to file for inspection
    with open(f"debug_prompt_{debug_id}.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # Encode for model
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # ---- Generate ----
    output_ids = model.generate(
        **encoded,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.convert_tokens_to_ids("<END>"),
    )[0]

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Strip prompt
    response = decoded[len(prompt):]
    response = response.split("<END>")[0].strip()

    print("\n📝 Raw model response:\n", response)
    return response

def extract_inside_ids(text):
    m = re.search(r"inside_ids:\s*\[(.*?)\]", text)
    if not m:
        return []
    return re.findall(r"'(.*?)'", m.group(1))

# ======================================================
# RUN TEST
# ======================================================
print("🚀 Running tests...")

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        item = json.loads(line)
        instr = item["instruction"]
        inp = item["input"]

        print(f"\n================= Test {i} =================")

        resp = generate_response(instr, inp, debug_id=i)
        inside = extract_inside_ids(resp)

        print("\n👉 Parsed inside_ids:", inside)
