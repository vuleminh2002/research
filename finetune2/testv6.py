import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TEST_FILE = "geocode_train_vary_test.jsonl"
MODEL_DIR = "tinyllama-geocode-lora_v6"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "geocode_test_predictions.jsonl"


# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
END_ID = tokenizer.convert_tokens_to_ids("<END>")


# -----------------------------
# Load base model in 8-bit
# -----------------------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# CRITICAL FIX: Resize base model to match tokenizer + LoRA vocab
base_model.resize_token_embeddings(len(tokenizer))


# -----------------------------
# Load LoRA adapter into base model
# -----------------------------
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

# Optional: Merge LoRA into the base model for faster inference
model = model.merge_and_unload()

model.eval()


# -----------------------------
# Helper: Generate output for one example
# -----------------------------
def generate_response(instruction, input_text, max_new_tokens=512):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=END_ID,   # STOP at <END>
            pad_token_id=END_ID,
            do_sample=False,       # deterministic
            temperature=0.0,
        )[0]

    text = tokenizer.decode(output_ids, skip_special_tokens=False)

    # Cut off at <END>
    if "<END>" in text:
        text = text.split("<END>")[0]

    # Keep only content after "### Response:"
    if "### Response:" in text:
        text = text.split("### Response:")[1].strip()

    return text


# -----------------------------
# Run inference on the test dataset
# -----------------------------
print("🔍 Running inference on test data...")

predictions = []

with open(TEST_FILE) as f:
    for line in tqdm(f, desc="Processing"):
        ex = json.loads(line)

        instruction = ex["instruction"]
        input_text = ex["input"]

        predicted = generate_response(instruction, input_text)

        predictions.append({
            "instruction": instruction,
            "input": input_text,
            "predicted_output": predicted
        })

# -----------------------------
# Save predictions
# -----------------------------
with open(OUTPUT_FILE, "w") as f:
    for p in predictions:
        f.write(json.dumps(p) + "\n")

print(f"\n✅ DONE! Predictions saved to: {OUTPUT_FILE}")
