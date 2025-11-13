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
# 1. Load dataset
# ============================================================
dataset = load_dataset("json", data_files={"train": "geocode_train_randomized.jsonl"})
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

# ============================================================
# 2. Load tokenizer + model
# ============================================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add <END>
tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
END_ID = tokenizer.convert_tokens_to_ids("<END>")
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Resize embeddings after adding new token
model.resize_token_embeddings(len(tokenizer))

# ============================================================
# 3. Apply LoRA
# ============================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ============================================================
# 4. Tokenization with label masking
# ============================================================
def tokenize_example(example):
    # Build full prompt
    instruction = example["instruction"].strip()
    inp = example["input"].strip()
    out = example["output"].strip()

    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n{out}\n<END>"
    )

    # Tokenize entire prompt
    full_tok = tokenizer(
        full_prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

    # Tokenize only the response part (without any specials)
    response_text = f"{out}\n<END>"
    response_ids = tokenizer(
        response_text,
        add_special_tokens=False
    )["input_ids"]

    input_ids = full_tok["input_ids"]

    # Find start index of response in the full sequence
    def find_subseq(full, sub):
        L = len(sub)
        for i in range(len(full) - L + 1):
            if full[i:i+L] == sub:
                return i
        return -1

    start = find_subseq(input_ids, response_ids)
    if start == -1:
        # fallback (rare)
        start = len(input_ids) - len(response_ids)

    # Build labels: mask everything before response and all padding
    labels = []
    for i, tok in enumerate(input_ids):
        if i < start:
            labels.append(-100)
        elif tok == tokenizer.pad_token_id:
            labels.append(-100)
        else:
            labels.append(tok)

    full_tok["labels"] = labels
    return full_tok
    

tokenized_train = train_ds.map(tokenize_example, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(tokenize_example, remove_columns=val_ds.column_names)

data_collator = default_data_collator

# ============================================================
# 5. Training configuration
# ============================================================
training_args = TrainingArguments(
    output_dir="tinyllama-geocode-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=3e-4,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=25,
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=200,
    save_steps=400,
    save_total_limit=3,
    report_to="none",
    optim="adamw_torch_fused",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=0,  # important for Windows
    max_grad_norm=1.0,
    prediction_loss_only=True,
)

# ============================================================
# 6. Trainer setup
# ============================================================
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False  # required for gradient checkpointing

def compute_metrics(_):
    return {}

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
# 7. Custom logging loop
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
        print(f"Eval loss = {loss:.4f} | Perplexity = {ppl:.2f}")

        writer.writerow([epoch+1, loss, ppl])
        f.flush()

# ============================================================
# 8. Save
# ============================================================
trainer.save_model("tinyllama-geocode-lora")
tokenizer.save_pretrained("tinyllama-geocode-lora")

print("✅ Training complete. Model saved to tinyllama-geocode-lora.")
print(f"📈 Metrics logged to {log_path}")
