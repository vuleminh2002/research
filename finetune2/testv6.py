import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "tinyllama-geocode-lora_final"
TEST_FILE = "geocode_train_vary_test.jsonl"
OUTPUT_FILE = "geocode_test_results.jsonl"

print("=" * 60)
print("🧪 GEOCODE MODEL TESTING")
print("=" * 60)

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================
print(f"\n🧠 Loading model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,  # For faster inference
)
model.eval()

print(f"✅ Model loaded")
print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# ============================================================
# LOAD TEST DATA
# ============================================================
print(f"\n📂 Loading test data from {TEST_FILE}...")
with open(TEST_FILE, "r") as f:
    test_data = [json.loads(line) for line in f]

print(f"✅ Loaded {len(test_data)} test examples")

# ============================================================
# INFERENCE FUNCTION
# ============================================================
def generate_response(instruction, input_text, max_new_tokens=1024):
    """Generate model prediction for given instruction and input."""
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (not the prompt)
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()

# ============================================================
# EXTRACT inside_ids FROM RESPONSE
# ============================================================
def extract_inside_ids(response):
    """Extract the inside_ids list from model response."""
    try:
        # Find "inside_ids:" in the response
        if "inside_ids:" not in response:
            return None
        
        # Extract the part after "inside_ids:"
        ids_part = response.split("inside_ids:")[1].strip()
        
        # Use eval to parse the list (safe since we control the input)
        # Remove any trailing text after the list
        if "]" in ids_part:
            ids_part = ids_part[:ids_part.index("]") + 1]
        
        inside_ids = eval(ids_part)
        return inside_ids
    except Exception as e:
        print(f"⚠️  Error extracting inside_ids: {e}")
        return None

# ============================================================
# EVALUATE PREDICTIONS
# ============================================================
def evaluate_prediction(predicted_ids, true_ids):
    """Calculate precision, recall, F1 for a single example."""
    if predicted_ids is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": False}
    
    pred_set = set(predicted_ids)
    true_set = set(true_ids)
    
    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = len(pred_set & true_set) / len(pred_set)
    
    if len(true_set) == 0:
        recall = 0.0
    else:
        recall = len(pred_set & true_set) / len(true_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    exact_match = pred_set == true_set
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    }

# ============================================================
# RUN TESTS
# ============================================================
print("\n" + "=" * 60)
print("🔍 RUNNING TESTS")
print("=" * 60)

results = []
all_metrics = []

for idx, example in enumerate(test_data, 1):
    print(f"\n{'=' * 60}")
    print(f"📝 TEST EXAMPLE {idx}/{len(test_data)}")
    print(f"{'=' * 60}")
    
    # Get ground truth
    true_output = example["output"].replace("</s>", "").strip()
    true_ids = extract_inside_ids(true_output)
    
    # Generate prediction
    print("\n⏳ Generating prediction...")
    predicted_output = generate_response(
        example["instruction"],
        example["input"]
    )
    predicted_ids = extract_inside_ids(predicted_output)
    
    # Evaluate
    metrics = evaluate_prediction(predicted_ids, true_ids)
    all_metrics.append(metrics)
    
    # Display results
    print("\n" + "-" * 60)
    print("🟦 INPUT (first 200 chars):")
    print("-" * 60)
    print(example["input"][:200] + "..." if len(example["input"]) > 200 else example["input"])
    
    print("\n" + "-" * 60)
    print("🤖 MODEL OUTPUT:")
    print("-" * 60)
    print(predicted_output)
    
    print("\n" + "-" * 60)
    print("🏷️  GROUND TRUTH:")
    print("-" * 60)
    print(true_output)
    
    print("\n" + "-" * 60)
    print("📊 EVALUATION:")
    print("-" * 60)
    print(f"Predicted IDs: {predicted_ids}")
    print(f"True IDs:      {true_ids}")
    print(f"Precision:     {metrics['precision']:.2%}")
    print(f"Recall:        {metrics['recall']:.2%}")
    print(f"F1 Score:      {metrics['f1']:.2%}")
    print(f"Exact Match:   {'✅ YES' if metrics['exact_match'] else '❌ NO'}")
    
    # Save result
    results.append({
        "example_id": idx,
        "instruction": example["instruction"],
        "input": example["input"],
        "predicted_output": predicted_output,
        "true_output": true_output,
        "predicted_ids": predicted_ids,
        "true_ids": true_ids,
        "metrics": metrics
    })

# ============================================================
# AGGREGATE METRICS
# ============================================================
print("\n" + "=" * 60)
print("📊 OVERALL RESULTS")
print("=" * 60)

avg_precision = sum(m["precision"] for m in all_metrics) / len(all_metrics)
avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
exact_match_count = sum(1 for m in all_metrics if m["exact_match"])
exact_match_rate = exact_match_count / len(all_metrics)

print(f"\nTotal Examples:    {len(test_data)}")
print(f"Average Precision: {avg_precision:.2%}")
print(f"Average Recall:    {avg_recall:.2%}")
print(f"Average F1 Score:  {avg_f1:.2%}")
print(f"Exact Match Rate:  {exact_match_rate:.2%} ({exact_match_count}/{len(test_data)})")

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n💾 Saving results to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

# Also save summary
summary_file = OUTPUT_FILE.replace(".jsonl", "_summary.json")
summary = {
    "total_examples": len(test_data),
    "metrics": {
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1,
        "exact_match_rate": exact_match_rate,
        "exact_match_count": exact_match_count
    },
    "per_example_metrics": all_metrics
}

with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Results saved to: {OUTPUT_FILE}")
print(f"✅ Summary saved to: {summary_file}")
print("\n" + "=" * 60)
print("🎉 TESTING COMPLETE!")
print("=" * 60 + "\n")
