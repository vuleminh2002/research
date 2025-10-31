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
import torch

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

# ✅ Modern quantization config (instead of load_in_8bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
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
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
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
        max_length=2048,  # ✅ Reduced for speed
    )

tokenized_train = train_ds.map(format_example, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(format_example, remove_columns=val_ds.column_names)

# ============================================================
# 5. Data collator
# ============================================================
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ============================================================
# 6. Training configuration
# ============================================================
training_args = TrainingArguments(
    output_dir="tinyllama-geocode-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # ✅ Faster updates
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    do_eval=True,  # ✅ Replaces deprecated evaluation_strategy
    eval_strategy="steps",  # For modern versions
    eval_steps=300,  # ✅ Less frequent evals
    save_steps=600,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ============================================================
# 7. Trainer setup
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# 8. Train and save
# ============================================================
torch.backends.cudnn.benchmark = True  # ✅ Slight speed boost
trainer.train()
model.save_pretrained("tinyllama-geocode-lora")
print("✅ Done! Adapter saved to tinyllama-geocode-lora")
