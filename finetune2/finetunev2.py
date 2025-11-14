import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "geocode_train_vary.jsonl"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "tinyllama-geocode-lora_final"
MAX_LENGTH = 2048

print("=" * 60)
print("🚀 GEOCODE MODEL TRAINING — USING </s> EOS")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


print(f"\n📂 Loading dataset: {DATA_FILE}")
raw_data = load_jsonl(DATA_FILE)
print(f"Total examples: {len(raw_data)}")

# Train/val split
train_raw, val_raw = train_test_split(raw_data, test_size=0.1, random_state=42)
print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")


# ============================================================
# TOKENIZER — USING NATIVE </s> AS EOS
# ============================================================
print("\n🧠 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Ensure PAD token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"EOS token: {tokenizer.eos_token} (ID {tokenizer.eos_token_id})")


# ============================================================
# TOKENIZATION + RESPONSE-ONLY LOSS MASKING
# ============================================================
RESPONSE_MARKER = "### Response:\n"
marker_ids = tokenizer(RESPONSE_MARKER, add_special_tokens=False)["input_ids"]

def find_subseq(main, sub):
    """Find subsequence index inside token list."""
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
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )

    ids = tok["input_ids"]

    # Find response start
    marker_idx = find_subseq(ids, marker_ids)

    if marker_idx == -1:
        # Should never happen, but fallback
        resp_start = max(0, len(ids) - 256)
    else:
        resp_start = marker_idx + len(marker_ids)

    # Build labels
    labels = []
    for i, t in enumerate(ids):
        if i < resp_start:
            labels.append(-100)          # Mask prompt
        else:
            labels.append(t)             # Learn response (incl. </s>)

    tok["labels"] = labels
    return tok


print("\n🔄 Tokenizing...")
train_ds = Dataset.from_dict({k: [ex[k] for ex in train_raw] for k in train_raw[0]}).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"],
)

val_ds = Dataset.from_dict({k: [ex[k] for ex in val_raw] for k in val_raw[0]}).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"],
)


# ============================================================
# DATA COLLATOR (CRITICAL!)
# ============================================================
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)


# ============================================================
# LOAD BASE MODEL (8-BIT)
# ============================================================
print("\n🧠 Loading base model (8-bit QLoRA)...")

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)


# ============================================================
# APPLY LORA
# ============================================================
print("\n🔧 Applying LoRA...")

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# ============================================================
# TRAINING ARGUMENTS
# ============================================================
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
    save_strategy="epoch",
    save_total_limit=2,

    bf16=use_bf16,
    fp16=not use_bf16,

    report_to="none",
    optim="adamw_torch_fused",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


# ============================================================
# TRAIN
# ============================================================
model.config.use_cache = False  # Required for gradient checkpointing

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
)

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# MERGE AND SAVE FINAL MODEL
# ============================================================
print("\n" + "=" * 60)
print("💾 SAVING MERGED MODEL")
print("=" * 60)

# Merge LoRA
model = model.merge_and_unload()

# Save final model + tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 TRAINING COMPLETE!")
print(f"📦 Model saved to: {OUTPUT_DIR}")
print("=" * 60)
