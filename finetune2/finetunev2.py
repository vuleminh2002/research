import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==============================================================
# CONFIG
# ==============================================================
DATA_FILE = "geocode_train_vary.jsonl"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "tinyllama-geocode-lora_FINAL"
MAX_LEN = 2048

RESPONSE_MARKER = "### Response:"    # <- NEW: No trailing newline

print("="*60)
print("🚀 FINAL TRAINING SCRIPT — TINYLLAMA GEOCODE")
print("="*60)


# ==============================================================
# LOAD DATA
# ==============================================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


raw = load_jsonl(DATA_FILE)
print(f"📂 Loaded {len(raw)} examples")

split = int(len(raw) * 0.9)
train_raw = raw[:split]
val_raw   = raw[split:]

print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")


# ==============================================================
# TOKENIZER
# ==============================================================
print("\n🧠 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# pad token fix
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"EOS token = {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")

# Tokenize the response marker ONCE
marker_ids = tokenizer(RESPONSE_MARKER, add_special_tokens=False)["input_ids"]


# ==============================================================
# SUBSTRING SEARCH (ROBUST AGAINST CHAT TEMPLATE WRAPS)
# ==============================================================
def find_marker(tokens, marker):
    """
    Safe subsequence search for the marker *anywhere* inside the
    chat-template-wrapped prompt.
    """
    m = len(marker)
    for i in range(len(tokens) - m + 1):
        if tokens[i:i+m] == marker:
            return i
    return -1


# ==============================================================
# TOKENIZATION + MASKING
# ==============================================================
def tokenize_example(ex):
    """
    Build Llama-chat style input with:
    ### Instruction
    ### Input
    ### Response
    Reasoning...
    """
    full_text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"{RESPONSE_MARKER}\n{ex['output']}"
    )

    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

    ids = enc["input_ids"]

    # find marker safely
    idx = find_marker(ids, marker_ids)

    if idx == -1:
        # This will *never* happen at training time with TinyLlama Chat.
        # But ensure safety:
        raise ValueError("❌ Marker not found in encode() — unexpected.")

    # response starts *after* the marker ids
    resp_start = idx + len(marker_ids)

    labels = []
    for i, tok in enumerate(ids):
        if i < resp_start:
            labels.append(-100)
        else:
            labels.append(tok)

    enc["labels"] = labels
    return enc


print("\n🔄 Tokenizing train dataset...")
train_ds = Dataset.from_dict({
    "instruction": [x["instruction"] for x in train_raw],
    "input":       [x["input"] for x in train_raw],
    "output":      [x["output"] for x in train_raw],
}).map(tokenize_example, remove_columns=["instruction", "input", "output"])

print("🔄 Tokenizing val dataset...")
val_ds = Dataset.from_dict({
    "instruction": [x["instruction"] for x in val_raw],
    "input":       [x["input"] for x in val_raw],
    "output":      [x["output"] for x in val_raw],
}).map(tokenize_example, remove_columns=["instruction", "input", "output"])


# ==============================================================
# DATA COLLATOR (CORRECTLY PADS LABELS WITH -100)
# ==============================================================
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)


# ==============================================================
# LOAD MODEL IN 8-BIT
# ==============================================================
print("\n🧠 Loading base model in 8-bit...")
bnb = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)


# ==============================================================
# APPLY LORA
# ==============================================================
print("\n🔧 Applying LoRA...")
lora = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()


# ==============================================================
# TRAINER + ARGS
# ==============================================================
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine",

    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",

    save_total_limit=2,
    report_to="none",

    bf16=use_bf16,
    fp16=not use_bf16,

    optim="adamw_torch_fused",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
)

print("\n================================================")
print("🚀 STARTING TRAINING")
print("================================================")

trainer.train()


# ==============================================================
# MERGE & SAVE
# ==============================================================
print("\n🔧 Merging LoRA into base model...")
merged = model.merge_and_unload()

print("💾 Saving final merged model...")
merged.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 TRAINING COMPLETE!")
print(f"Model saved to: {OUTPUT_DIR}")
