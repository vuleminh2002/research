import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import sys

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR = "tinyllama-geocode-lora_v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🧠 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("✅ Loaded tokenizer from LoRA directory.")
except Exception as e:
    print("❌ Failed to load tokenizer from LoRA directory.")
    print("⚠ Falling back to base tokenizer (NOT recommended)")
    tokenizer = AutoTokenizer.from_pretrained(BASE)

# Ensure END token exists
if "<END>" not in tokenizer.get_vocab():
    print("⚠ WARNING: <END> token missing — adding it.")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})

END_ID = tokenizer.convert_tokens_to_ids("<END>")
tokenizer.pad_token = tokenizer.eos_token

print("\n🧠 Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Resize embeddings BEFORE loading LoRA
base.resize_token_embeddings(len(tokenizer))

print("🧠 Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base,
    MODEL_DIR,
)

model = model.to(DEVICE)
model.eval()


# ============================================================
#  GENERATION FUNCTION (stops at <END>)
# ============================================================
def generate_response(instruction, input_text, max_new_tokens=512):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    print("\n🔍 Full prompt provided to the model:\n" + prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=END_ID     # STOP when model outputs <END>
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=False)

    # Cut at <END>
    if "<END>" in decoded:
        decoded = decoded.split("<END>")[0]

    # Remove everything before ### Response:
    if "### Response:" in decoded:
        decoded = decoded.split("### Response:")[1].strip()

    print("\n📝 Raw model response:\n", decoded)
    return decoded


# ============================================================
#  TESTING WITH ONE SAMPLE
# ============================================================
# Load a sample from your dataset
sample = None
try:
    with open("sample_input.json", "r") as f:
        sample = json.load(f)
except:
    print("\n⚠ No sample_input.json found. Using a hardcoded test case.\n")
    sample = {
        "instruction": "Classify each candidate business as inside or outside the given rectangular range based on its latitude and longitude. Output reasoning for each candidate, then list inside_ids.",
        "input": """Rectangle:
  top_left: (40.1430, -86.3922)
  bottom_right: (39.4482, -85.6975)
Candidates:
  j2T-O-fk5E4BjcoUC1tRIA: (39.6213, -86.1590)
  MfK5DeKjnneIDZTUkUKP8g: (39.9782, -86.1286)
  sDP2_XUcyN9W2cSV8OCp9g: (39.8330, -86.2456)
  1Tku3PDAmSOoJeR80tuZqA: (39.9229, -86.2306)
  fr5URHkQU59sdka9ShjHJw: (39.9184, -86.2248)
  fci-hYpdFDMKgLLqRXobUw: (39.9276, -86.0937)
  pk7E-3G7Ij39h9HifNz0TA: (39.8535, -85.9926)
  cM-mwZrOPbeXRQqrXDU1_Q: (39.9520, -86.0409)
  rZVznzcoBnX2DOYwPdE5NA: (39.8639, -86.3985)
  oT5Bidkfa7cGOp1806ryXQ: (39.9137, -86.1391)
  LMahsH7Iy9d3MT3ez_CmCQ: (39.6132, -86.0835)
  fqKPLGsHisaCtNHFVyvUzg: (39.6101, -86.3736)
"""
    }

instruction = sample["instruction"]
input_text = sample["input"]

response = generate_response(instruction, input_text)
print("\n👉 Final Extracted Response:\n", response)
