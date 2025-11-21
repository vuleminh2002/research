import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------- CONFIG ----------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR   = "/research/finetune2/tinyllama-geocode-lora_s2"
OUT_DIR    = "/research/finetune2/tinyllama-geocode-merged-bf16"

print("🧠 Loading base model in BF16…")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},  # H100
)

print("🔧 Loading LoRA adapter…")
model = PeftModel.from_pretrained(
    base,
    LORA_DIR,
    is_trainable=False,
    local_files_only=True,
)

print("🔗 Merging LoRA into base weights…")
model = model.merge_and_unload()        # after this, no PEFT

print("💾 Saving merged model to:", OUT_DIR)
model.save_pretrained(OUT_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUT_DIR)

print("✅ Done. Merged model saved.")
