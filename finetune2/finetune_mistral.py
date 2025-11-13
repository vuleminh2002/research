import os, torch, gc, csv, math
gc.collect()
torch.cuda.empty_cache()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 1. Load dataset
# ============================================================
dataset = load_dataset("json", data_files={"train": "geocode_train_randomized.jsonl"})
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

# ============================================================
# 2. Load Mistral tokenizer + base model
# ============================================================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    use_fast=True
)

# Add <END> token
tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
END_ID = tokenizer.convert_tokens_to_ids("<END>")
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 3. Load model with QLoRA optimized for A100
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # BEST for A100
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.resize_token_embeddings(len(tokenizer))

# Enable flash attention (A100 supports it)
model.config.use_flash_attention = True


# ============================================================
# 4. Apply LoRA (correct modules for Mistral)
# ============================================================
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# 5. Tokenizer → mask labels outside response
# ============================================================
MAX_LEN = 4096  # A100 40GB safe; 80GB can use 8192

def tokenize(example):
    instruction = example["instruction"].strip()
    inp = example["input"].strip()
    out = example["output"].strip()

    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n{out}\n<END>"
    )

    # Tokenize full sequence
    tok = tokenizer(
        full_prompt,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )

    # Tokens for response only
    resp_text = f"{out}\n<END>"
    resp_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]

    ids = tok["input_ids"]

    # Find where response starts
    def find_subseq(full, sub):
        L = len(sub)
        for i in range(len(full) - L + 1):
            if full[i:i+L] == sub:
                return i
        return -1

    start = find_subseq(ids, resp_ids)
    if start == -1:
        start = len(ids) - len(resp_ids)

    # Build labels
    labels = []
    for i, t in enumerate(ids):
        if i < start or t == tokenizer.pad_token_id:
            labels.append(-100)
        else:
            labels.append(t)

    tok["labels"] = labels
    return tok


train_tok = train_ds.map(tokenize, remove_columns=train_ds.column_names)
val_tok = val_ds.map(tokenize, remove_columns=val_ds.column_names)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    max_length=MAX_LEN
)

# ============================================================
# 6. TrainingArguments (optimized for A100)
# ============================================================
training_args = TrainingArguments(
    output_dir="mistral-geocode-lora",

    # Batch size
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   # Effective batch size = 16

    # Training schedule
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    # Logging / saving / evaluation — MUST MATCH
    logging_steps=20,
    evaluation_strategy="steps",     # <--- REQUIRED
    eval_steps=200,
    save_strategy="steps",           # <--- MUST MATCH evaluation_strategy
    save_steps=200,
    save_total_limit=3,

    # Hardware optimization for A100
    bf16=True,                       # A100 supports bfloat16
    optim="adamw_torch_fused",
    gradient_checkpointing=True,

    # For using best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",
)

model.enable_input_require_grads()
model.config.use_cache = False  # Required with gradient checkpointing


# ============================================================
# 7. Trainer
# ============================================================
def compute_metrics(_):
    return {}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# 8. Training loop + logging
# ============================================================
log_file = "training_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eval_loss", "perplexity"])

    for epoch in range(training_args.num_train_epochs):
        print(f"\n🚀 Starting Epoch {epoch+1}")
        trainer.train()

        m = trainer.evaluate()
        loss = m["eval_loss"]
        ppl = math.exp(loss) if loss < 20 else float("inf")
        print(f"Eval Loss: {loss:.4f}   |   Perplexity: {ppl:.2f}")

        writer.writerow([epoch+1, loss, ppl])
        f.flush()

# ============================================================
# 9. Save output
# ============================================================
trainer.save_model("mistral-geocode-lora")
tokenizer.save_pretrained("mistral-geocode-lora")

print("✅ Training Complete.")
