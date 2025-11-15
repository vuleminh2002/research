import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_FILE = "geocode_train_vary.jsonl"
OUTPUT_DIR = "tinyllama-geocode-lora_s1"
MAX_LENGTH = 2048

print("="*60)
print("🚀 FINAL TRAINING SCRIPT — STABLE INSTRUCTION FORMAT")
print("="*60)


# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

raw_data = load_jsonl(TRAIN_FILE)
print(f"📂 Loaded {len(raw_data)} examples")

split = int(0.9 * len(raw_data))
train_raw = raw_data[:split]
val_raw   = raw_data[split:]

print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# TinyLlama has no PAD → use EOS as PAD
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"EOS token = '{tokenizer.eos_token}' (id {tokenizer.eos_token_id})")


# ============================================================
# TOKENIZATION — OPTION A (Robust, No Marker Required)
# ============================================================

def tokenize_example(ex):
    # Build full prompt
    full_text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}"
    )

    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = enc["input_ids"]

    # Compute token index where OUTPUT begins (safe + precise)
    output_char_start = full_text.index(ex["output"])
    output_token_start = len(
        tokenizer(full_text[:output_char_start], add_special_tokens=False)["input_ids"]
    )

    # Mask prompt tokens
    labels = [
        tok if i >= output_token_start else -100
        for i, tok in enumerate(input_ids)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


print("🔄 Tokenizing...")

# Your environment does NOT support Dataset.from_list
train_ds = Dataset.from_dict({
    "instruction": [ex["instruction"] for ex in train_raw],
    "input": [ex["input"] for ex in train_raw],
    "output": [ex["output"] for ex in train_raw],
}).map(tokenize_example, remove_columns=["instruction", "input", "output"])

val_ds = Dataset.from_dict({
    "instruction": [ex["instruction"] for ex in val_raw],
    "input": [ex["input"] for ex in val_raw],
    "output": [ex["output"] for ex in val_raw],
}).map(tokenize_example, remove_columns=["instruction", "input", "output"])

dataset = DatasetDict({"train": train_ds, "val": val_ds})


# ============================================================
# COLLATOR
# ============================================================

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)


# ============================================================
# LOAD MODEL + APPLY LORA
# ============================================================

print("🧠 Loading base model in 8-bit…")

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
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
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
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_steps=400,
    eval_strategy="epoch",
    bf16=use_bf16,
    fp16=not use_bf16,
    optim="adamw_torch_fused",
    report_to="none",
)


# ============================================================
# TRAIN
# ============================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=collator,
)

print("\n🚀 Starting training…")
trainer.train()

print("\n🔧 Merging LoRA weights…")
merged = model.merge_and_unload()
merged.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Training complete!")
