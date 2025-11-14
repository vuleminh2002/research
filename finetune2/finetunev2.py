import torch, gc, math, csv
gc.collect()
torch.cuda.empty_cache()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 1. LOAD DATASET
# ============================================================
dataset = load_dataset("json", data_files={"train": "geocode_train_vary.jsonl"})
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

# ============================================================
# 2. TOKENIZER — ADD <END> AND MAKE IT EOS + PAD
# ============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
STUDENT_MODEL_DIR = "tinyllama-geocode-lora_v6"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Add <END> as an extra token (if it's not already there)
tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})

# Configure <END> as EOS and PAD token
tokenizer.eos_token = "<END>"
tokenizer.pad_token = "<END>"
END_ID = tokenizer.convert_tokens_to_ids("<END>")

# Save tokenizer config
tokenizer.save_pretrained(STUDENT_MODEL_DIR)

# ============================================================
# 3. LOAD MODEL (AFTER TOKENIZER IS FINAL)
# ============================================================
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# Resize embeddings for new special token(s)
model.resize_token_embeddings(len(tokenizer))

model.config.eos_token_id = END_ID
model.config.pad_token_id = END_ID

# ============================================================
# 4. APPLY QLoRA PARAMETERS
# ============================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 5. TOKENIZATION WITH RESPONSE-ONLY MASKING
# ============================================================

RESPONSE_MARKER = "### Response:\n"
response_marker_ids = tokenizer(RESPONSE_MARKER, add_special_tokens=False)["input_ids"]

def find_subsequence(main, sub):
    """Return index where `sub` starts in `main`, or -1 if not found."""
    n, m = len(main), len(sub)
    for i in range(n - m + 1):
        if main[i : i + m] == sub:
            return i
    return -1

def tokenize_example(example):
    instruction = example["instruction"].strip()
    inp = example["input"].strip()
    out = example["output"].strip()  # should already end with <END>

    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"{RESPONSE_MARKER}{out}"
    )

    # Tokenize without padding; truncation only
    full_tok = tokenizer(
        full_prompt,
        truncation=True,
        max_length=2048,
    )

    ids = full_tok["input_ids"]

    # Find where the response starts (right after "### Response:\n")
    marker_index = find_subsequence(ids, response_marker_ids)
    if marker_index == -1:
        resp_start = 0
    else:
        resp_start = marker_index + len(response_marker_ids)

    # Create labels: mask everything before resp_start
    labels = []
    for i, tok in enumerate(ids):
        if i < resp_start:
            labels.append(-100)
        else:
            labels.append(tok)

    full_tok["labels"] = labels
    return full_tok

train_tok = train_ds.map(tokenize_example, remove_columns=train_ds.column_names)
val_tok = val_ds.map(tokenize_example, remove_columns=val_ds.column_names)

# ============================================================
# 6. DATA COLLATOR — PADS INPUTS *AND* LABELS
# ============================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,  # optional, but good for tensor cores
)

# ============================================================
# 7. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir=STUDENT_MODEL_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=3e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    fp16=True,
    logging_steps=25,

    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=400,
    save_total_limit=2,

    report_to="none",
    optim="adamw_torch_fused",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    prediction_loss_only=True,
)

# ============================================================
# 8. TRAINER SETUP
# ============================================================
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False  # important for gradient checkpointing

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# 9. TRAIN
# ============================================================
trainer.train()

# Write eval metrics to CSV
log_path = f"{STUDENT_MODEL_DIR}/training_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eval_step", "eval_loss", "perplexity"])

    for log in trainer.state.log_history:
        if "eval_loss" in log:
            loss = log["eval_loss"]
            epoch = log.get("epoch")
            step = log.get("step")
            ppl = math.exp(loss) if loss < 20 else float("inf")
            writer.writerow([epoch, step, loss, ppl])

metrics = trainer.evaluate()
loss = metrics["eval_loss"]
ppl = math.exp(loss) if loss < 20 else float("inf")
print(f"\nFINAL EVAL — loss: {loss:.4f}, ppl: {ppl:.2f}")

# ============================================================
# 10. SAVE MODEL + TOKENIZER
# ============================================================
trainer.save_model(STUDENT_MODEL_DIR)
tokenizer.save_pretrained(STUDENT_MODEL_DIR)

print("\n✅ TRAINING COMPLETE — Saved to tinyllama-geocode-lora_v6")
