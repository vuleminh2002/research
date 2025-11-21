import json
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR   = "tinyllama-geocode-lora_s2"
TEST_FILE  = "geocode_train_vary_test.jsonl"

MAX_NEW_TOKENS = 512


print("=" * 70)
print("🚀 FAST INFERENCE + METRICS — TinyLlama + LoRA")
print("=" * 70)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id


# ============================================================
# LOAD MODEL IN 4-BIT AND APPLY LORA
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
)

model = PeftModel.from_pretrained(base, LORA_DIR)
model.eval()


# ============================================================
# GENERATION (TPS)
# ============================================================

@torch.inference_mode()
def generate_response(prompt: str):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    prompt_len = encoded["input_ids"].shape[1]

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=True
    ):
        torch.cuda.synchronize()
        start = time.time()

        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
        )

        torch.cuda.synchronize()
        end = time.time()

    gen_ids = output[0][prompt_len:]
    gen_len = len(gen_ids)
    elapsed = end - start
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    tps = gen_len / elapsed if elapsed > 0 else 0
    return decoded, gen_len, elapsed, tps


# ============================================================
# PARSE PREDICTED inside_ids FROM MODEL OUTPUT
# ============================================================

def extract_inside_ids(text):
    """
    Extract inside_ids list from model output.
    Model output ends with: inside_ids: ['id1', 'id2', ...]
    """
    if "inside_ids" not in text:
        return []

    try:
        # Get substring after inside_ids:
        part = text.split("inside_ids")[1]
        # Expect format: ": ['id1', 'id2']"
        start = part.find("[")
        end = part.find("]")
        if start == -1 or end == -1:
            return []

        inside_str = part[start+1 : end].strip()
        if inside_str == "":
            return []

        # Split entries
        ids = [s.strip().strip("'").strip('"') for s in inside_str.split(",")]
        return [x for x in ids if len(x) > 0]
    except:
        return []


# ============================================================
# METRICS FUNCTIONS
# ============================================================

def compute_metrics(pred, gold):
    """
    pred: set([candidate_ids])
    gold: set([candidate_ids])
    """
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 1.0

    return precision, recall,
