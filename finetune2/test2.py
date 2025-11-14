import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = "tinyllama-geocode-lora_v3"
DATA_FILE = "geocode_train_vary.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------
# 1. Load tokenizer
# -------------------------------------------------------------
print("🧠 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -------------------------------------------------------------
# 2. Load merged model (NOT LoRA)
# -------------------------------------------------------------
print("🧠 Loading merged student model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# -------------------------------------------------------------
# 3. Load first 4 training samples
# -------------------------------------------------------------
print("📄 Loading training examples...")
examples = []
with open(DATA_FILE, "r") as f:
    for _ in range(4):
        examples.append(json.loads(next(f)))

print(f"Loaded {len(examples)} examples.\n")

# -------------------------------------------------------------
# 4. Build prompt exactly like training
# -------------------------------------------------------------
def build_prompt(example):
    instruction = example["instruction"].strip()
    inp = example["input"].strip()

    prompt = (
        f"{tokenizer.bos_token}"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )
    return prompt

# -------------------------------------------------------------
# 5. Run inference
# -------------------------------------------------------------
for idx, ex in enumerate(examples):
    print("=" * 80)
    print(f"🧪 TEST SAMPLE #{idx+1}")
    print("=" * 80)

    prompt = build_prompt(ex)

    print("\n🔍 PROMPT SENT TO MODEL:\n")
    print(prompt)
    print("-" * 80)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    print(f"🔢 Prompt tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.replace(prompt, "").strip()

    print("\n📝 RAW MODEL OUTPUT:\n")
    print(response)
    print("-" * 80)

    # Extract inside_ids
    inside = []
    for line in response.split("\n"):
        if line.lower().startswith("inside_ids"):
            try:
                arr = line.split(":", 1)[1].strip()
                inside = json.loads(arr.replace("'", '"'))
            except:
                inside = []
            break

    print(f"👉 Parsed inside_ids: {inside}\n")
