import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------------
# 2. Load model
# -------------------------------------------------------------
print("🧠 Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)

# If LoRA adapter exists, load it
try:
    print("🔧 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
except:
    print("⚠ No LoRA adapter found — using full model.")
    model = base

model.eval()

# -------------------------------------------------------------
# 3. Load FIRST 4 samples from the training dataset
# -------------------------------------------------------------
print("📄 Loading training examples...")
examples = []
with open(DATA_FILE, "r") as f:
    for _ in range(4):
        examples.append(json.loads(next(f)))

print(f"Loaded {len(examples)} examples.\n")

# -------------------------------------------------------------
# 4. Format prompt EXACTLY like training
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
# 5. Run inference on all 4 training examples
# -------------------------------------------------------------
for idx, ex in enumerate(examples):
    print("=" * 80)
    print(f"🧪 TEST SAMPLE #{idx + 1}")
    print("=" * 80)

    prompt = build_prompt(ex)
    print("\n🔍 PROMPT SENT TO MODEL:\n")
    print(prompt)
    print("\n" + "-" * 80)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    print(f"🔢 Prompt token length = {inputs['input_ids'].shape[1]}")

    # -------------------------------
    # Generate
    # -------------------------------
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,       # deterministic for debugging
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.replace(prompt, "").strip()

    print("\n📝 RAW MODEL OUTPUT:\n")
    print(response)
    print("\n" + "-" * 80)

    # ----------------------------------------------------------
    # Extract inside_ids
    # ----------------------------------------------------------
    extracted = []

    for line in response.split("\n"):
        if line.lower().startswith("inside_ids"):
            try:
                arr = line.split(":", 1)[1].strip()
                extracted = json.loads(arr.replace("'", '"'))
            except:
                extracted = []
            break

    print(f"👉 Parsed inside_ids: {extracted}\n")
