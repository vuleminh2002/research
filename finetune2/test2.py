import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "tinyllama-geocode-lora_s1"
TEST_FILE = "geocode_train_vary_test.jsonl"
MAX_NEW_TOKENS = 1024

print("=" * 60)
print("🔍 GEOCODE MODEL INFERENCE — TOKEN USAGE DEBUG")
print("=" * 60)


# ================================================================
# Load tokenizer & model
# ================================================================
print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

eos_id = tokenizer.eos_token_id
print(f"EOS token: {tokenizer.eos_token}  (ID {eos_id})")

print("\n🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
)
model.eval()
print("✅ Model loaded\n")


# ================================================================
# Inference helper — with EOS + token usage debugging
# ================================================================
def run_model(instruction, input_text):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
            return_dict_in_generate=True,
        )

    seq = out.sequences[0]
    gen_ids = seq[prompt_len:]
    gen_list = gen_ids.tolist()

    # RAW decode
    raw_text = tokenizer.decode(gen_list, skip_special_tokens=False)

    # CLEAN decode (trim at </s>)
    clean_text = raw_text.split("</s>")[0].strip()

    # Token usage information
    eos_pos = gen_list.index(eos_id) if eos_id in gen_list else None
    used_tokens = eos_pos + 1 if eos_pos is not None else len(gen_list)

    return {
        "clean": clean_text,
        "raw": raw_text,
        "prompt_len": prompt_len,
        "gen_len": len(gen_list),
        "eos_pos": eos_pos,
        "used_tokens": used_tokens,
        "gen_ids": gen_list,
    }


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

    res = run_model(ex["instruction"], ex["input"])

    # ===============================
    # TOKEN USAGE REPORT
    # ===============================
    print("\n🔢 TOKEN USAGE")
    print("-" * 60)
    print(f"Prompt tokens:       {res['prompt_len']}")
    print(f"Generated tokens:     {res['gen_len']}")
    if res["eos_pos"] is not None:
        print(f"EOS found at index:   {res['eos_pos']}")
        print(f"Tokens used until EOS:{res['used_tokens']}")
    else:
        print("❌ EOS NOT reached — model used full max_new_tokens!")

    # ===============================
    # INPUT / OUTPUT
    # ===============================
    print("\n🟦 INPUT:")
    print("-" * 60)
    print(ex["input"])

    print("\n🤖 RAW MODEL OUTPUT:")
    print("-" * 60)
    print(res["raw"])

    print("\n🤖 CLEAN MODEL OUTPUT:")
    print("-" * 60)
    print(res["clean"])

    print("\n🏷️ TRUE OUTPUT:")
    print("-" * 60)
    print(ex["output"].replace("</s>", "").strip())

    print("\n")


print("=" * 60)
print("✅ DONE — ALL TEST EXAMPLES COMPLETE")
print("=" * 60)
