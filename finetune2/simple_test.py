import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from difflib import SequenceMatcher

# ============================================================
# 1. Config
# ============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "tinyllama-geocode-lora"  # your LoRA checkpoint
DATA_FILE = "geocode_train_randomized2.jsonl"  # your 10-sample test file
MAX_SAMPLES = 10

# ============================================================
# 2. Load Model + Tokenizer
# ============================================================
print("🧠 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# ============================================================
# 3. Helper: Compute similarity score
# ============================================================
def similarity(a, b):
    return round(SequenceMatcher(None, a.strip(), b.strip()).ratio(), 3)

# ============================================================
# 4. Evaluate model on test data
# ============================================================
print(f"📄 Loading up to {MAX_SAMPLES} test samples from {DATA_FILE}...")
with open(DATA_FILE) as f:
    data = [json.loads(line) for line in f][:MAX_SAMPLES]

total_sim = 0
for i, sample in enumerate(data, 1):
    instruction = sample["instruction"]
    input_text = sample["input"]
    ground_truth = sample["output"]

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    print(f"\n{'🧩' * 10}\n🧩 SAMPLE {i}\n{'🧩' * 10}")
    print(f"📥 INPUT:\n{input_text}\n")

    # ------------------------------------------------------------
    # Generate model response
    # ------------------------------------------------------------
    result = pipe(
        prompt,
        max_new_tokens=1500,
        temperature=0.2,
        top_p=0.9,
        do_sample=False,
    )[0]["generated_text"]

    # extract only generated part
    model_output = result.split("### Response:")[-1].strip()

    # 🧹 Quick cleanup: cut after outside_ids
    if "outside_ids" in model_output:
        model_output = model_output.split("outside_ids:")  # split into parts
        model_output = model_output[0] + "outside_ids:" + model_output[-1].split("\n")[0]

    # Optional: also stop at the next '###' section if model repeats the prompt
    model_output = model_output.split("###")[0].strip()
    # show raw output
    print("🤖 MODEL OUTPUT:\n")
    print(model_output)
    
    # ------------------------------------------------------------
    # Show ground truth
    # ------------------------------------------------------------
    print(f"\n🎯 GROUND TRUTH:\n")
    print(ground_truth)

    # ------------------------------------------------------------
    # Compare to ground truth
    # ------------------------------------------------------------
    score = similarity(model_output, ground_truth)
    total_sim += score
    print(f"\n📊 Similarity to ground truth: {score}")

# ============================================================
# 5. Summary
# ============================================================
avg_score = total_sim / len(data)
print(f"\n✅ Done. Average similarity score across {len(data)} samples: {avg_score:.3f}")
