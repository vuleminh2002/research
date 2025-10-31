import torch, gc
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
    r=32,                   # ✅ higher rank = more expressive updates
    lora_alpha=64,          # ✅ balanced scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # ✅ cover all attention projections
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
    per_device_train_batch_size=8,            # ✅ was 2 → now 8 (use GPU VRAM fully)
    gradient_accumulation_steps=2,            # ✅ keeps total batch size manageable
    num_train_epochs=8,                       # ✅ longer training improves pattern learning
    learning_rate=3e-4,                       # ✅ slightly higher; faster convergence
    lr_scheduler_type="constant",             # ✅ no decay—keeps learning signal strong
    warmup_ratio=0.03,                        # ✅ small warmup prevents early spikes
    fp16=True,                                # ✅ fast training on modern GPUs
    logging_steps=25,                         # ✅ more frequent logs
    eval_strategy="steps",                    # ✅ evaluate every few hundred steps
    eval_steps=200,                           # ✅ earlier feedback loops
    save_steps=400,                           # ✅ frequent checkpoints (in case of crash)
    save_total_limit=3,                       # ✅ keeps storage clean
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch_fused",                # ✅ fused optimizer = faster
    dataloader_num_workers=4,                 # ✅ better I/O throughput
    max_grad_norm=1.0,  
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
