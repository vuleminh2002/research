import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TEST_FILE = "geocode_train_vary_test.jsonl"
MODEL_DIR = "tinyllama-geocode-lora_v6"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "geocode_test_predictions.jsonl"

print("=" * 60)
print("🚀 GEOCODE MODEL INFERENCE")
print("=" * 60)

# -----------------------------
# Load tokenizer
# -----------------------------
print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
END_ID = tokenizer.convert_tokens_to_ids("<END>")

print(f"✅ Tokenizer loaded")
print(f"   <END> token ID: {END_ID}")
print(f"   Vocab size: {len(tokenizer)}")

# -----------------------------
# Load base model in 8-bit
# -----------------------------
print("\n🧠 Loading base model in 8-bit...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# CRITICAL: Resize embeddings to match tokenizer
base_model.resize_token_embeddings(len(tokenizer))
print(f"✅ Base model loaded and resized")

# -----------------------------
# Load LoRA adapter and merge
# -----------------------------
print("\n🔧 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

print("🔗 Merging LoRA weights...")
model = model.merge_and_unload()
model.eval()
print("✅ Model ready for inference")

# -----------------------------
# Inference function
# -----------------------------
def generate_response(instruction, input_text, max_new_tokens=1024, debug=False):
    """
    Generate model response for given instruction and input.
    
    Args:
        instruction: Task instruction
        input_text: Input data
        max_new_tokens: Maximum tokens to generate
        debug: If True, print debug information
    
    Returns:
        Generated response text (without <END> token)
    """
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=END_ID,
            pad_token_id=END_ID,
            do_sample=False,
        )

    # Extract only the generated tokens (not the input prompt)
    generated_ids = output_ids[0][input_length:]
    
    if debug:
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Input length: {input_length} tokens")
        print(f"   Generated length: {len(generated_ids)} tokens")
        print(f"   <END> token ({END_ID}) in generated: {END_ID in generated_ids.tolist()}")
        print(f"   Last 10 token IDs: {generated_ids[-10:].tolist()}")
    
    # Decode the generated tokens
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    if debug:
        print(f"   '<END>' in decoded text: {'<END>' in text}")
        print(f"   Last 100 chars: ...{text[-100:]}")
    
    # Remove <END> token if present
    if "<END>" in text:
        text = text.split("<END>")[0]
    
    return text.strip()


# -----------------------------
# Run inference on test dataset
# -----------------------------
print("\n" + "=" * 60)
print("🔍 RUNNING INFERENCE ON TEST DATASET")
print("=" * 60 + "\n")

predictions = []
total_examples = 0

# Count total examples first
with open(TEST_FILE) as f:
    total_examples = sum(1 for _ in f)

print(f"📊 Found {total_examples} test examples\n")

# Process each example
with open(TEST_FILE) as f:
    for idx, line in enumerate(tqdm(f, total=total_examples, desc="Processing"), 1):
        ex = json.loads(line)

        instruction = ex["instruction"]
        input_text = ex["input"]
        true_output = ex.get("output", "").replace("<END>", "").strip()

        # Generate prediction (debug=True for first example only)
        predicted = generate_response(instruction, input_text, debug=(idx == 1))

        # Print results
        print("\n" + "=" * 60)
        print(f"📝 EXAMPLE {idx}/{total_examples}")
        print("=" * 60)
        
        print("\n🟦 INPUT:")
        print("-" * 60)
        print(input_text[:300] + "..." if len(input_text) > 300 else input_text)
        
        print("\n🤖 MODEL OUTPUT:")
        print("-" * 60)
        print(predicted)
        
        print("\n🏷️  TRUE OUTPUT:")
        print("-" * 60)
        print(true_output[:300] + "..." if len(true_output) > 300 else true_output)
        print()

        # Collect prediction
        predictions.append({
            "instruction": instruction,
            "input": input_text,
            "predicted_output": predicted,
            "true_output": true_output
        })

# -----------------------------
# Save predictions to file
# -----------------------------
print("\n" + "=" * 60)
print("💾 SAVING PREDICTIONS")
print("=" * 60)

with open(OUTPUT_FILE, "w") as f:
    for p in predictions:
        f.write(json.dumps(p) + "\n")

print(f"\n✅ DONE!")
print(f"📁 Predictions saved to: {OUTPUT_FILE}")
print(f"📊 Total examples processed: {len(predictions)}")
print("\n" + "=" * 60 + "\n")