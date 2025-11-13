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
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 1. LOAD DATASET
# ============================================================
dataset = load_dataset("json", data_files={"train": "geocode_train_randomized.jsonl"})
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

# ============================================================
# 2. LOAD TOKENIZER + ADD <END>
# ============================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
END_ID = tokenizer.convert_tokens_to_ids("<END>")
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 3. LOAD MODEL + QLoRA 8-bit
# ============================================================
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Resize embeddings for <END>
model.resize_token_embeddings(len(tokenizer))

# ============================================================
# 4. APPLY LoRA
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
# 5. TOKENIZATION & LABEL MASKING
# ============================================================
def tokenize_example(example):
    instruction = example["instruction"].strip()
    inp = example["input"].strip()
    out = example["output"].strip()  # now contains only reasoning + inside_ids

    # FULL PROMPT (Chat-style format)
    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n{out}\n<END>"
    )

    # Tokenize whole prompt → input_ids + pad
    full_tok = tokenizer(
        full_prompt,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

    # Tokens of response ONLY
    resp_text = f"{out}\n<END>"
    resp_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]

    ids = full_tok["input_ids"]

    # Locate response start index in full prompt
    def find_subseq(full, sub):
        L = len(sub)
        for i in range(len(full) - L + 1):
            if full[i:i+L] == sub:
                return i
        return -1

    start = find_subseq(ids, resp_ids)
    if start == -1:
        # Rare fallback: assume response at end
        start = len(ids) - len(resp_ids)

    # MASK LABELS → Only response tokens are learned
    labels = []
    for i, t in enumerate(ids):
        if i < start:  
            labels.append(-100)         # mask instruction + input
        elif t == tokenizer.pad_token_id:
            labels.append(-100)         # mask padding
        else:
            labels.append(t)            # learn response

    full_tok["labels"] = labels
    return full_tok


tokenized_train = train_ds.map(tokenize_example, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(tokenize_example, remove_columns=val_ds.column_names)

data_collator = default_data_collator

# ============================================================
# 6. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir="tinyllama-geocode-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=3e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",

    fp16=True,
    logging_steps=25,

    eval_strategy="steps",
    eval_steps=200,
    save_steps=400,

    save_total_limit=3,
    report_to="none",

    optim="adamw_torch_fused",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    prediction_loss_only=True,
)

# ============================================================
# 7. TRAINER
# ============================================================
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False  # required for checkpointing

def compute_metrics(_): return {}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    compute_metrics=compute_metrics,
)

# ============================================================
# 8. CUSTOM LOGGING LOOP
# ============================================================
log_path = "training_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eval_loss", "perplexity"])

    for epoch in range(training_args.num_train_epochs):
        print(f"\n🚀 Epoch {epoch+1}")
        trainer.train()

        metrics = trainer.evaluate()
        loss = metrics["eval_loss"]
        ppl = math.exp(loss) if loss < 20 else float("inf")

        print(f"Eval Loss = {loss:.4f} | Perplexity = {ppl:.2f}")
        writer.writerow([epoch+1, loss, ppl])
        f.flush()

# ============================================================
# 9. SAVE MODEL
# ============================================================
trainer.save_model("tinyllama-geocode-lora")
tokenizer.save_pretrained("tinyllama-geocode-lora")

print("\n✅ Training complete.")
