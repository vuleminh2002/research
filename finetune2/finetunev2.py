import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from sklearn.model_selection import train_test_split

# ===============================================================
# CONFIG
# ===============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE  = "geocode_train_vary.jsonl"
OUTPUT_DIR = "tinyllama-geocode-lora_final"
MAX_LENGTH = 2048

print("="*60)
print("🚀 GEOCODE MODEL TRAINING — SINGLE FILE MODE")
print("="*60)

# ===============================================================
# LOAD DATA
# ===============================================================
print("\n📂 Loading dataset:", DATA_FILE)
with open(DATA_FILE, "r") as f:
    data_raw = [json.loads(line) for line in f]

print(f"Total examples: {len(data_raw)}")

# Split automatically: 90% train / 10% val
train_raw, val_raw = train_test_split(
    data_raw, test_size=0.1, random_state=42
)

print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")

# ===============================================================
# TOKENIZER
# ===============================================================
print("\n🧠 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Use model's natural EOS as pad token
tokenizer.pad_token = tokenizer.eos_token
print(f"EOS token: {tokenizer.eos_token} (ID {tokenizer.eos_token_id})")

# ===============================================================
# TOKENIZATION + RESPONSE MASKING
# ===============================================================
RESPONSE_MARKER = "### Response:\n"
MARKER_IDS = tokenizer(RESPONSE_MARKER, add_special_tokens=False)["input_ids"]

def find_subseq(main, sub):
    n, m = len(main), len(sub)
    for i in range(n - m + 1):
        if main[i:i+m] == sub:
            return i
    return -1

def tokenize_example(ex):
    full_text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"{RESPONSE_MARKER}{ex['output']}"
    )

    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = tok["input_ids"]
    attn = tok["attention_mask"]

    # Locate response marker
    marker_idx = find_subseq(input_ids, MARKER_IDS)
    if marker_idx == -1:
        raise RuntimeError(
            "❌ Response marker not found in tokenized sequence.\n"
            "Your prompt is too long — increase MAX_LENGTH or shorten inputs."
        )

    resp_start = marker_idx + len(MARKER_IDS)

    labels = [
        tok_id if i >= resp_start else -100
        for i, tok_id in enumerate(input_ids)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }

print("\n🔄 Tokenizing train...")
train_ds = Dataset.from_list(train_raw).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"]
)

print("🔄 Tokenizing val...")
val_ds = Dataset.from_list(val_raw).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"]
)

# ===============================================================
# SIMPLE CAUSAL LM DATA COLLATOR
# ===============================================================
def causal_collator(batch):
    return tokenizer.pad(
        batch,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# ===============================================================
# LOAD MODEL + APPLY LORA
# ===============================================================
print("\n🧠 Loading model in 8-bit...")

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ===============================================================
# TRAINER SETUP
# ===============================================================
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine",

    logging_steps=10,
    eval_strategy="epoch",
    save_steps=400,
    save_total_limit=2,

    bf16=use_bf16,
    fp16=not use_bf16,

    report_to="none",
    optim="adamw_torch_fused",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

model.config.use_cache = False

trainer = Trainer(
    model=model,
    args=args,
    data_collator=causal_collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)

trainer.train()

# ===============================================================
# MERGE + SAVE STANDALONE MODEL
# ===============================================================
print("\nMerging LoRA weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 Training COMPLETE!")
print(f"📦 Saved to {OUTPUT_DIR}")
print("="*60)
