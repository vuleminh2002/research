import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ============================================================
# 1. Paths and model setup
# ============================================================
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "tinyllama-geocode-lora"
data_file = "geocode_train_randomized.jsonl"

print("🚀 Loading base model + LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(model, adapter_dir)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    temperature=0.0,
    max_new_tokens=512,
    do_sample=False,
)

# ============================================================
# 2. Load first 10 samples
# ============================================================
print(f"📂 Loading first 10 samples from {data_file} ...")
with open(data_file, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for _, line in zip(range(10), f)]
print(f"✅ Loaded {len(examples)} samples.\n")

# ============================================================
# 3. Helper function to extract IDs
# ============================================================
def extract_ids(text, key):
    """
    Extracts IDs from model output or ground truth lines like:
    inside_ids: ['abc', 'def'] or inside_ids: []
    """
    match = re.search(rf"{key}:\s*\[([^\]]*)\]", text)
    if not match:
        return []
    ids_raw = match.group(1)
    ids = re.findall(r"'([^']+)'", ids_raw)
    return ids

# ============================================================
# 4. Run inference and collect predictions
# ============================================================
id_level_true = []
id_level_pred = []

for i, ex in enumerate(tqdm(examples, desc="Evaluating samples")):
    prompt = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )
    output = pipe(prompt, max_new_tokens=512)[0]["generated_text"]
    response_text = output.split("### Response:")[-1].strip()

    # Extract IDs
    gt_inside = extract_ids(ex["output"], "inside_ids")
    pred_inside = extract_ids(response_text, "inside_ids")

    # Collect all unique IDs for evaluation
    all_ids = set(gt_inside + pred_inside)
    for cid in all_ids:
        id_level_true.append(1 if cid in gt_inside else 0)
        id_level_pred.append(1 if cid in pred_inside else 0)

    # Print detailed result
    print(f"\n{'='*80}")
    print(f"🧩 SAMPLE {i+1}")
    print(f"📥 INPUT:\n{ex['input'][:500]}...\n")
    print(f"🤖 FULL MODEL OUTPUT:\n{response_text}\n")
    print(f"🎯 GROUND TRUTH:\n{ex['output'][:700]}\n")
    print(f"🔍 Parsed inside_ids (pred): {pred_inside}")
    print(f"✅ Parsed inside_ids (true): {gt_inside}")
    print(f"{'='*80}\n")

# ============================================================
# 5. Compute metrics
# ============================================================
if id_level_true:
    acc = accuracy_score(id_level_true, id_level_pred)
    prec = precision_score(id_level_true, id_level_pred, zero_division=0)
    rec = recall_score(id_level_true, id_level_pred, zero_division=0)
    f1 = f1_score(id_level_true, id_level_pred, zero_division=0)

    print("\n📊 === Overall Evaluation (first 10 samples) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
else:
    print("⚠️ No valid inside_ids found to evaluate.")
