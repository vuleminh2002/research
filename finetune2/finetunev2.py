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
VAL_SPLIT = 0.1

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

print("=" * 60)
print("🚀 LORA TRAINING — TinyLlama Geocode (Response-only Loss, EOS-aware)")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

raw_data = load_jsonl(TRAIN_FILE)
print(f"📂 Loaded {len(raw_data)} examples")

split = int((1.0 - VAL_SPLIT) * len(raw_data))
train_raw = raw_data[:split]
val_raw   = raw_data[split:]

print(f"📊 Train: {len(train_raw)}, Val: {len(val_raw)}")

# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# TinyLlama has no PAD → use EOS as PAD
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EOS_TOKEN = tokenizer.eos_token
EOS_ID = tokenizer.eos_token_id
print(f"EOS token = '{EOS_TOKEN}' (id {EOS_ID})")
print(f"PAD token = '{tokenizer.pad_token}' (id {tokenizer.pad_token_id})")

# ============================================================
# TOKENIZATION (PROMPT + RESPONSE, LOSS ONLY ON RESPONSE)
# ============================================================

def normalize_output_text(output_text: str) -> str:
    """
    Ensure each output ends with a single EOS token string (</s> for TinyLlama).
    If the data already has it, keep it. Otherwise, append.
    """
    text = output_text.rstrip()
    if not text.endswith(EOS_TOKEN):
        text = text + EOS_TOKEN
    return text

def tokenize_example(ex):
    """
    Build:
      prompt_text = "### Instruction: ... ### Input: ... ### Response:\n"
      answer_text = normalized output (guaranteed to end with </s>)

    Then:
      input_ids = prompt_ids + answer_ids
      labels    = -100 for prompt, real ids for answer
    """
    instruction = ex["instruction"]
    input_ = ex["input"]
    output = normalize_output_text(ex["output"])

    prompt_text = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_}\n\n"
        f"### Response:\n"
    )
    answer_text = output

    # Tokenize separately to know exactly where answer starts in token space
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )
    answer_enc = tokenizer(
        answer_text,
        add_special_tokens=False,
    )

    input_ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
    attention_mask = [1] * len(input_ids)

    # Truncate to MAX_LENGTH if needed (keep left side, cut tail)
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]

    # Index where answer tokens start
    answer_start = len(prompt_enc["input_ids"])

    # If truncation cut off the entire answer region (rare), then no labels
    if answer_start >= len(input_ids):
        labels = [-100] * len(input_ids)
    else:
        labels = []
        for i, tok in enumerate(input_ids):
            if i < answer_start:
                labels.append(-100)  # ignore prompt in loss
            else:
                labels.append(tok)   # train only on answer (response)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("🔄 Tokenizing train/val splits...")

train_ds = Dataset.from_dict({
    "instruction": [ex["instruction"] for ex in train_raw],
    "input": [ex["input"] for ex in train_raw],
    "output": [ex["output"] for ex in train_raw],
}).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"]
)

val_ds = Dataset.from_dict({
    "instruction": [ex["instruction"] for ex in val_raw],
    "input": [ex["input"] for ex in val_raw],
    "output": [ex["output"] for ex in val_raw],
}).map(
    tokenize_example,
    remove_columns=["instruction", "input", "output"]
)

dataset = DatasetDict({"train": train_ds, "val": val_ds})
print(dataset)

# Small sanity check: decode one example to verify boundaries
print("\n🔍 Sanity check example:")
sample = dataset["train"][0]
ids = sample["input_ids"]
labels = sample["labels"]

decoded_full = tokenizer.decode(ids, skip_special_tokens=False)
print("Full text:")
print(decoded_full)

# Show where supervised region starts
supervised_tokens = [tok for tok, lab in zip(ids, labels) if lab != -100]
print("\nSupervised (answer) region decode:")
print(tokenizer.decode(supervised_tokens, skip_special_tokens=False))

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

print("\n🧠 Loading base model in 8-bit…")

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
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
    save_total_limit=3,
    evaluation_strategy="epoch",    # correct arg name
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

print("\n💾 Saving LoRA adapters (NO MERGE)...")
# This saves ONLY LoRA weights (adapter_model.safetensors + adapter_config.json)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Training complete!")
print("✅ LoRA adapters saved to:", OUTPUT_DIR)
print("   Use base TinyLlama + this folder for fast 4-bit inference.")
