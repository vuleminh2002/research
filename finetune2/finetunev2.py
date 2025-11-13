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
split_ds = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split_ds["train"], split_ds["test"]

# ============================================================
# 2. Load tokenizer and model
# ============================================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Properly add <END> as a special token
special_tokens = {"additional_special_tokens": ["<END>"]}
tokenizer.add_special_tokens(special_tokens)

# Use EOS as pad
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Resize embeddings after adding <END>
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
END_TOKEN = "<END>"

def format_example(example):
    instruction = f"### Instruction:\n{example['instruction'].strip()}\n\n"
    input_part = f"### Input:\n{example['input'].strip()}\n\n"
    response_part = f"### Response:\n{example['output'].strip()}\n{END_TOKEN}"

    full_prompt = instruction + input_part + response_part

    # Tokenize entire prompt
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        padding="max_length",
        max_length=2048,
    )
    input_ids = tokenized["input_ids"]

    # Tokenize only the response (no extra specials)
    response_ids = tokenizer(
        response_part,
        add_special_tokens=False,
    )["input_ids"]

    # Find where response sequence begins
    def find_subsequence(full, part):
        for i in range(len(full) - len(part) + 1):
            if full[i:i+len(part)] == part:
                return i
        return -1

    start = find_subsequence(input_ids, response_ids)
    if start == -1:
        # Fallback if alignment fails (rare, e.g. truncation)
        start = len(input_ids) - len(response_ids)

    # Build labels:
    # - everything before response_start -> -100 (ignored)
    # - padding tokens -> -100 (ignored)
    # - response tokens -> learn normally
    labels = []
    pad_id = tokenizer.pad_token_id
    for idx, tok_id in enumerate(input_ids):
        if idx < start:
            labels.append(-100)
        else:
            if tok_id == pad_id:
                labels.append(-100)
            else:
                labels.append(tok_id)

    tokenized["labels"] = labels
    return tokenized

tokenized_train = train_ds.map(format_example, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(format_example, remove_columns=val_ds.column_names)

data_collator = default_data_collator

# ============================================================
# 5. Training configuration
# ============================================================
training_args = TrainingArguments(
    output_dir="tinyllama-geocode-lorav2",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=3e-4,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=400,
    save_total_limit=3,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch_fused",
    dataloader_num_workers=0,  # safer on Windows
    max_grad_norm=1.0,
)

# ============================================================
# 6. Trainer setup
# ============================================================
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False

def compute_metrics(eval_pred):
    # Trainer will pass (predictions, labels) by default, but we only care about loss via trainer.evaluate()
    # so we keep this minimal; actual metrics logged manually below.
    return {}

if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics,
    )

    # ========================================================
    # 7. Custom logging loop
    # ========================================================
    log_file = "training_log.csv"
    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "eval_loss", "perplexity"])

        for epoch in range(int(training_args.num_train_epochs)):
            print(f"\n🚀 Epoch {epoch + 1}/{int(training_args.num_train_epochs)}")
            trainer.train()

            metrics = trainer.evaluate()
            loss = metrics["eval_loss"]
            perplexity = math.exp(loss) if loss < 20 else float("inf")
            print(f"✅ Epoch {epoch + 1}: Eval loss = {loss:.4f}, Perplexity = {perplexity:.2f}")
            writer.writerow([epoch + 1, loss, perplexity])
            csvfile.flush()

    # ========================================================
    # 8. Save model + tokenizer
    # ========================================================
    save_dir = "tinyllama-geocode-lorav2"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"✅ Done! Model + LoRA + tokenizer saved to {save_dir}")
    print(f"📈 Metrics logged to {log_file}")
