import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "tinyllama-geocode-lora_s1"      # merged model directory
TEST_FILE = "geocode_train_vary_test.jsonl"  # test dataset with 4 examples
MAX_NEW_TOKENS = 1024

print("=" * 60)
print("🔍 GEOCODE MODEL INFERENCE — WITH EOS DEBUGGING")
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
# Inference helper — with EOS debugging
# ================================================================
def run_model(instruction, input_text):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=False,
        )

    # Full sequence returned
    seq = output.sequences[0]

    # Extract only new generated tokens
    gen_ids = seq[input_len:]

    # -------------------------------
    # 🔍 DEBUG INFORMATION
    # -------------------------------
    print("\n\n===============================")
    print("🔍 GENERATION DEBUG")
    print("===============================")
    print(f"Generated tokens: {len(gen_ids)}")
    print(f"First 10 tokens: {gen_ids[:10].tolist()}")
    print(f"Last 10 tokens:  {gen_ids[-10:].tolist()}")

    gen_list = gen_ids.tolist()
    if eos_id in gen_list:
        eos_pos = gen_list.index(eos_id)
        print(f"✅ EOS STOP detected at position {eos_pos}")
    else:
        print("❌ No EOS detected — model used full max_new_tokens!")

    # Raw decode (no trimming)
    raw_text = tokenizer.decode(gen_list, skip_special_tokens=False)

    # Clean output (trim at </s>)
    clean_text = raw_text.split("</s>")[0].strip()

    return clean_text, raw_text


# ================================================================
# Load test dataset
# ================================================================
print(f"📂 Loading test file: {TEST_FILE}")
examples = [json.loads(line) for line in open(TEST_FILE)]
print(f"📊 Total test examples: {len(examples)}\n")


# ================================================================
# Run tests
# ================================================================
for idx, ex in enumerate(examples, start=1):

    print("\n" + "=" * 60)
    print(f"📝 TEST EXAMPLE {idx}/{len(examples)}")
    print("=" * 60)

    instruction = ex["instruction"]
    input_text = ex["input"]
    true_output = ex["output"].replace("</s>", "").strip()

    # Run model
    clean_pred, raw_pred = run_model(instruction, input_text)

    # --------------------------
    # PRINT RESULTS
    # --------------------------

    print("\n🟦 INPUT:")
    print("-" * 60)
    print(input_text)

    print("\n🤖 RAW MODEL OUTPUT (no trimming):")
    print("-" * 60)
    print(raw_pred)

    print("\n🤖 CLEAN MODEL OUTPUT (trimmed):")
    print("-" * 60)
    print(clean_pred)

    print("\n🏷️ TRUE OUTPUT:")
    print("-" * 60)
    print(true_output)

    print("\n")


print("=" * 60)
print("✅ DONE — ALL TEST EXAMPLES COMPLETE")
print("=" * 60)
