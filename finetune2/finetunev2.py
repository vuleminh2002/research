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


# ===============================================================
# CONFIG
# ===============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_FILE = "geocode_train_vary.jsonl"
OUTPUT_DIR = "tinyllama-geocode-lora_FINAL"
MAX_LENGTH = 2048
RESPONSE_MARKER = "### Response:\n"


print("="*60)
print("🚀 GEOCODE RANGE CLASSIFIER — FINAL TRAINING SCRIPT")
print("="*60)


# ===============================================================
# LOAD DATASET
# ===============================================================

def load_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(json.loads(line))
    return out


print(f"\n📂 Loading dataset: {TRAIN_FILE}")
raw_data = load_jsonl(TRAIN_FILE)
print(f"Total examples: {len(raw_data)}")

# 90/10 split
split_idx = int(0.9 * len(raw_data))
train_raw = raw_data[:split_idx]
val_raw   = raw_data[split_idx:]

print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")


# ===============================================================
# TOKENIZER
# ===============================================================

print("\n🧠 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"EOS token: {tokenizer.eos_token} (ID {tokenizer.eos_token_id})")


# ===============================================================
# TOKENIZATION — FIXED WITH TRUE TOKEN-LEVEL MARKER SEARCH
# ===============================================================

def find_subsequence(tokens, marker):
    """
    Returns the first index where `marker` appears inside `tokens`.
    Token-level safe search — avoids SentencePiece boundary errors.
    """
    for i in range(len(tokens) - len(marker) + 1):
        if tokens[i:i+len(marker)] == marker:
            return i
    return -1


def tokenize_example(ex):
    """
    Builds:
    ### Instruction:
    ...
    ### Input:
    ...
    ### Response:
    <output>
    
    Then masks everything before the response.
    """
    # Construct full text
    full_text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"{RESPONSE_MARKER}{ex['output']}"
    )

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Tokenize the RESPONSE_MARKER only
    marker_ids = tokenizer(RESPONSE_MARKER, add_special_tokens=False)["input_ids"]

    # Find marker correctly in token space
    idx = find_subsequence(input_ids, marker_ids)

    if idx == -1:
        # Should never happen, but fallback
        resp_start = int(0.6 * len(input_ids))
    else:
        resp_start = idx + len(marker_ids)

    # Create labels: mask prompt, learn response only
    labels = []
    for i, tok in enumerate(input_ids):
        if i < resp_start:
            labels.append(-100)
        else:
            labels.append(tok)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("\n🔄 Tokenizing train...")
train_ds = Dataset.from_list(train_raw).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"],
)

print("🔄 Tokenizing val...")
val_ds = Dataset.from_list(val_raw).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"],
)

dataset = DatasetDict({"train": train_ds, "val": val_ds})


# ===============================================================
# DATA COLLATOR — CORRECT LABEL PADDING
# ===============================================================

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)


# ===============================================================
# LOAD MODEL + APPLY LORA
# ===============================================================

print("\n🧠 Loading base model...")

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model)

print("🔧 Applying LoRA...")
lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# ===============================================================
# TRAINER ARGS
# ===============================================================

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
    logging_steps=10,
    save_steps=400,
    eval_strategy="epoch",
    bf16=use_bf16,
    fp16=not use_bf16,
    report_to="none",
    optim="adamw_torch_fused",
    load_best_model_at_end=True,
)


# ===============================================================
# TRAIN
# ===============================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=collator,
)

print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)

trainer.train()


# ===============================================================
# MERGE LORA & SAVE FINAL MODEL
# ===============================================================

print("\n🔧 Merging LoRA weights...")
merged = model.merge_and_unload()

print("💾 Saving merged model...")
merged.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 TRAINING COMPLETE — MODEL SAVED!")
print(f"📦 Output: {OUTPUT_DIR}")
