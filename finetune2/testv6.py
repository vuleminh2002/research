import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "tinyllama-geocode-lora_final"       # your merged model
TEST_FILE = "geocode_train_vary_test.jsonl"      # your 4 test examples
MAX_NEW_TOKENS = 1024

print("=" * 60)
print("🔍 GEOCODE MODEL INFERENCE — USING </s>")
print("=" * 60)


# ================================================================
# Load tokenizer & model
# ================================================================
print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

eos_id = tokenizer.eos_token_id
print(f"EOS token: {tokenizer.eos_token} (ID {eos_id})")

print("\n🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
)
model.eval()

print("✅ Model loaded\n")


# ================================================================
# Helper: run inference for a single example
# ================================================================
def run_model(instruction, input_text):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = encoded["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
        )

    generated_ids = output[0][input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Truncate at first EOS if it appears
    if "</s>" in text:
        text = text.split("</s>")[0]

    return text.strip()


# ================================================================
# Load test dataset
# ================================================================
print(f"📂 Loading test file: {TEST_FILE}")
examples = [json.loads(line) for line in open(TEST_FILE)]
print(f"📊 Total test examples: {len(examples)}\n")


# ================================================================
# Run inference for each example
# ================================================================
for idx, ex in enumerate(examples, start=1):
    print("\n" + "=" * 60)
    print(f"📝 TEST EXAMPLE {idx}/{len(examples)}")
    print("=" * 60)

    instruction = ex["instruction"]
    input_text = ex["input"]
    true_output = ex["output"].replace("</s>", "").strip()

    # Run model
    pred = run_model(instruction, input_text)

    # Print results
    print("\n🟦 INPUT:")
    print("-" * 60)
    print(input_text)

    print("\n🤖 MODEL OUTPUT:")
    print("-" * 60)
    print(pred)

    print("\n🏷️ TRUE OUTPUT:")
    print("-" * 60)
    print(true_output)

    print("\n")


print("=" * 60)
print("✅ DONE — ALL TEST EXAMPLES COMPLETE")
print("=" * 60)
