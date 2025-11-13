import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from difflib import SequenceMatcher

# ============================================================
# 1. Config
# ============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "tinyllama-geocode-lora"  # your LoRA checkpoint
DATA_FILE = "geocode_train_randomized5.jsonl"  # your 10-sample test file
MAX_SAMPLES = 1

# ============================================================
# 2. Load Model + Tokenizer
# ============================================================
print("🧠 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)  # Comment out to test base model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# ============================================================
# 3. Helper functions
# ============================================================
def similarity(a, b):
    return round(SequenceMatcher(None, a.strip(), b.strip()).ratio(), 3)

def extract_lists(text):
    """Extract inside_ids and outside_ids lists from text"""
    inside_ids = []
    outside_ids = []
    
    # Try to find inside_ids
    if "inside_ids:" in text:
        inside_part = text.split("inside_ids:")[1]
        inside_line = inside_part.split("\n")[0].strip()
        try:
            # Try to parse as list
            if "[" in inside_line and "]" in inside_line:
                inside_str = inside_line.split("[")[1].split("]")[0]
                if inside_str.strip():
                    inside_ids = [x.strip().strip('"\'') for x in inside_str.split(",")]
        except:
            pass
    
    # Try to find outside_ids
    if "outside_ids:" in text:
        outside_part = text.split("outside_ids:")[1]
        outside_line = outside_part.split("\n")[0].strip()
        try:
            # Try to parse as list
            if "[" in outside_line and "]" in outside_line:
                outside_str = outside_line.split("[")[1].split("]")[0]
                if outside_str.strip():
                    outside_ids = [x.strip().strip('"\'') for x in outside_str.split(",")]
        except:
            pass
    
    return inside_ids, outside_ids

def compare_lists(model_inside, model_outside, truth_inside, truth_outside):
    """Compare extracted lists and return detailed scores"""
    
    # Convert to sets for comparison
    model_inside_set = set(model_inside)
    model_outside_set = set(model_outside)
    truth_inside_set = set(truth_inside)
    truth_outside_set = set(truth_outside)
    
    # Calculate precision, recall, F1 for inside_ids
    inside_correct = len(model_inside_set & truth_inside_set)
    
    # Handle perfect match when both are empty
    if len(model_inside_set) == 0 and len(truth_inside_set) == 0:
        inside_precision, inside_recall, inside_f1 = 1.0, 1.0, 1.0
    else:
        inside_precision = inside_correct / len(model_inside_set) if model_inside_set else 0
        inside_recall = inside_correct / len(truth_inside_set) if truth_inside_set else 0
        inside_f1 = 2 * inside_precision * inside_recall / (inside_precision + inside_recall) if (inside_precision + inside_recall) > 0 else 0
    
    # Calculate precision, recall, F1 for outside_ids  
    outside_correct = len(model_outside_set & truth_outside_set)
    
    # Handle perfect match when both are empty
    if len(model_outside_set) == 0 and len(truth_outside_set) == 0:
        outside_precision, outside_recall, outside_f1 = 1.0, 1.0, 1.0
    else:
        outside_precision = outside_correct / len(model_outside_set) if model_outside_set else 0
        outside_recall = outside_correct / len(truth_outside_set) if truth_outside_set else 0
        outside_f1 = 2 * outside_precision * outside_recall / (outside_precision + outside_recall) if (outside_precision + outside_recall) > 0 else 0
    
    # Overall accuracy
    total_correct = inside_correct + outside_correct
    total_predicted = len(model_inside_set) + len(model_outside_set)
    total_truth = len(truth_inside_set) + len(truth_outside_set)
    
    overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
    overall_recall = total_correct / total_truth if total_truth > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        'inside_precision': round(inside_precision, 3),
        'inside_recall': round(inside_recall, 3),
        'inside_f1': round(inside_f1, 3),
        'outside_precision': round(outside_precision, 3),
        'outside_recall': round(outside_recall, 3),
        'outside_f1': round(outside_f1, 3),
        'overall_precision': round(overall_precision, 3),
        'overall_recall': round(overall_recall, 3),
        'overall_f1': round(overall_f1, 3)
    }

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
        max_new_tokens=3000,
        temperature=0.2,
        top_p=0.9,
        do_sample=False,
    )[0]["generated_text"]

    # Show complete raw model output (everything)
    print("🤖 COMPLETE MODEL OUTPUT:\n")
    print(result)
    print("\n" + "="*80 + "\n")
    
    # Use the complete raw result for extraction (no early splitting)
    model_output = result
    
    # ------------------------------------------------------------
    # Show ground truth
    # ------------------------------------------------------------
    print(f"\n🎯 GROUND TRUTH:\n")
    print(ground_truth)

    # ------------------------------------------------------------
    # Extract and compare lists
    # ------------------------------------------------------------
    model_inside, model_outside = extract_lists(model_output)
    truth_inside, truth_outside = extract_lists(ground_truth)
    
    print(f"\n📋 EXTRACTED LISTS:")
    print(f"Model inside_ids:  {model_inside}")
    print(f"Truth inside_ids:  {truth_inside}")
    print(f"Model outside_ids: {model_outside}")
    print(f"Truth outside_ids: {truth_outside}")
    
    # Compare lists and get detailed metrics
    metrics = compare_lists(model_inside, model_outside, truth_inside, truth_outside)
    
    print(f"\n📊 DETAILED METRICS:")
    print(f"Inside  - Precision: {metrics['inside_precision']}, Recall: {metrics['inside_recall']}, F1: {metrics['inside_f1']}")
    print(f"Outside - Precision: {metrics['outside_precision']}, Recall: {metrics['outside_recall']}, F1: {metrics['outside_f1']}")
    print(f"Overall - Precision: {metrics['overall_precision']}, Recall: {metrics['overall_recall']}, F1: {metrics['overall_f1']}")

    # ------------------------------------------------------------
    # Compare raw text similarity (for reference)
    # ------------------------------------------------------------
    text_score = similarity(model_output, ground_truth)
    total_sim += metrics['overall_f1']  # Use F1 score instead of text similarity
    print(f"\n� Text similarity: {text_score}")
    print(f"🎯 F1 Score: {metrics['overall_f1']}")

# ============================================================
# 5. Summary
# ============================================================
avg_f1_score = total_sim / len(data)
print(f"\n✅ Done. Average F1 score across {len(data)} samples: {avg_f1_score:.3f}")
print(f"📈 This measures how well the model extracts inside/outside ID lists compared to ground truth.")
