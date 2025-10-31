import torch, gc, math, csv
gc.collect()
torch.cuda.empty_cache()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
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
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

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
# 4. Tokenization function
# ============================================================
def format_example(example):
    prompt = (
        f"### Instruction:\n{example['instruction'].strip()}\n\n"
        f"### Input:\n{example['input'].strip()}\n\n"
        f"### Response:\n{example['output'].strip()}"
    )
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=2048,
    )

tokenized_train = train_ds.map(format_example, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(format_example, remove_columns=val_ds.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ============================================================
# 5. Compute metrics (eval_loss + perplexity)
# ============================================================
def compute_metrics(eval_pred):
    loss = eval_pred.metrics["eval_loss"]
    perplexity = math.exp(loss) if loss < 20 else float("inf")
    print(f"\n📊 Eval loss: {loss:.4f} | Perplexity: {perplexity:.2f}")
    return {"eval_loss": loss, "perplexity": perplexity}

# ============================================================
# 6. Training configuration
# ============================================================
training_args = TrainingArguments(
    output_dir="tinyllama-geocode-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
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
    dataloader_num_workers=4,
    max_grad_norm=1.0,
)

# ============================================================
# 7. Trainer setup
# ============================================================
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False
torch.backends.cudnn.benchmark = True

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# 8. Custom logger for eval loss + perplexity
# ============================================================
log_file = "training_log.csv"
with open(log_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["step", "eval_loss", "perplexity"])

    # Train loop with eval tracking
    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\n🚀 Epoch {epoch + 1}/{int(training_args.num_train_epochs)}")
        trainer.train()

        metrics = trainer.evaluate()
        loss = metrics["eval_loss"]
        perplexity = math.exp(loss) if loss < 20 else float("inf")
        print(f"✅ Epoch {epoch + 1}: Eval loss = {loss:.4f}, Perplexity = {perplexity:.2f}")

        writer.writerow([epoch + 1, loss, perplexity])
        csvfile.flush()

# ============================================================
# 9. Save model
# ============================================================
model.save_pretrained("tinyllama-geocode-lora")
print("✅ Done! Adapter saved to tinyllama-geocode-lora")
print(f"📈 All eval metrics logged to {log_file}")
