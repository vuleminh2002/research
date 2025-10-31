import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ============================================================
# 1. Load model and tokenizer
# ============================================================
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "tinyllama-geocode-lora"  # folder where your fine-tuned model is saved

print("🧠 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# ============================================================
# 2. Load test data (only input, no gold output)
# ============================================================
test_sample = {
    "instruction": "Classify each candidate business as inside or outside the given rectangular range based on its latitude and longitude. Output reasoning for each candidate, then list final inside_ids and outside_ids.",
    "input": "Rectangle:\n  top_left: (32.3418, -111.0662)\n  bottom_right: (32.1943, -110.9186)\nCandidates:\n  VhXfYBVHepqvpcLnX07XNg: (32.2202, -110.8598)\n  coR0dFgGOU7fVku6xDdH-A: (32.3202, -110.9757)\n  HDuR_Dtb5xLQyTbz53DJ9Q: (32.4276, -111.0796)\n  oZtaNtFTJkpfPG0hXc98Xw: (32.1332, -111.0297)\n  EILOpH5vO8vO2zGHXr11hA: (32.2506, -110.8907)\n  gKFyQ2Fc88JsQi9M6xiWfA: (32.2289, -110.9928)\n  wa_bwyY57etHjtJ2Fw0E3g: (32.2209, -110.8789)\n  VwCR9uFMaDbGhZuxtB_1rw: (32.4443, -110.9718)\n  zSnauQCNDfXyltOH_e1d8w: (32.2350, -110.8964)\n  caGpaptBP4BKJkpH5W6cvA: (32.2525, -110.9437)\n  sFQTyTiBqhTpSvyyB73iYA: (32.2893, -110.9752)\n  gD2_HqjOwttxj0S4v_v60Q: (32.2271, -110.9502)"
}

# ============================================================
# 3. Format prompt for generation
# ============================================================
prompt = f"### Instruction:\n{test_sample['instruction']}\n\n### Input:\n{test_sample['input']}\n\n### Response:\n"

# ============================================================
# 4. Generate model response
# ============================================================
print("🚀 Generating prediction...\n")
output = pipe(
    prompt,
    max_new_tokens=1200,      # allow enough reasoning space
    temperature=0.2,          # keep deterministic
    top_p=0.9,
    do_sample=False,
)[0]["generated_text"]

# Extract only model's new text after the prompt
generated_part = output.split("### Response:")[-1].strip()

print("🤖 MODEL OUTPUT:\n")
print(generated_part)
print("\n✅ Done.")
