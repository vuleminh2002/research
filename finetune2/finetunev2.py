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
dataset = load_dataset("json", data_files={"train": "geocode_train_vary.jsonl"})
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

# ============================================================
# 2. LOAD TOKENIZER FIRST — ADD <END> BEFORE MODEL LOAD
# ============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
STUDENT_MODEL_DIR = "tinyllama-geocode-lora_v3"   # <<<<<<<< UPDATED HERE

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Add <END> BEFORE loading model ⇨ prevents vocab mismatch
SPECIAL_TOKENS = {"additional_special_tokens": ["<END>"]}
tokenizer.add_special_tokens(SPECIAL_TOKENS)

END_ID = tokenizer.convert_tokens_to_ids("<END>")
tokenizer.pad_token = tokenizer.eos_token  # required for batching

# Save tokenizer into v3 folder
tokenizer.save_pretrained(STUDENT_MODEL_DIR)

# ============================================================
# 3. LOAD MODEL — AFTER TOKENIZER IS FINALIZED
# ============================================================
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# Resize embeddings NOW (before LoRA)
model.resize_token_embeddings(len(tokenizer))

# ============================================================
# 4. APPLY QLoRA
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
# 5. TOKENIZATION WITH RESPONSE-ONLY LABELING
# ============================================================
def tokenize_example(example):
    instruction = example["instruction"].strip()
    inp = example["input"].strip()
    out = example["output"].strip()

    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n{out}\n<END>"
    )

    full_tok = tokenizer(
        full_prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

    resp_text = f"{out}\n<END>"
    resp_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]

    ids = full_tok["input_ids"]

    def find_subseq(main, sub):
        L = len(sub)
        for i in range(len(main) - L + 1):
            if main[i:i+L] == sub:
                return i
        return -1

    start_idx = find_subseq(ids, resp_ids)
    if start_idx == -1:
        start_idx = len(ids) - len(resp_ids)  # fallback

    labels = []
    for i, t in enumerate(ids):
        if i < start_idx or t == tokenizer.pad_token_id:
            labels.append(-100)
        else:
            labels.append(t)

    full_tok["labels"] = labels
    return full_tok


train_tok = train_ds.map(tokenize_example, remove_columns=train_ds.column_names)
val_tok = val_ds.map(tokenize_example, remove_columns=val_ds.column_names)

data_collator = default_data_collator

# ============================================================
# 6. TRAINING ARGUMENTS — UPDATED FOR v3 MODEL DIR
# ============================================================
training_args = TrainingArguments(
    output_dir=STUDENT_MODEL_DIR,        # <<<<<<<< UPDATED
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=3e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",

    fp16=True,
    logging_steps=25,

    eval_strategy="steps",
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
# 7. TRAINER
# ============================================================
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# 8. CUSTOM TRAINING LOOP WITH LOGGING
# ============================================================
log_path = f"{STUDENT_MODEL_DIR}/training_log.csv"  # <<<<<<<< UPDATED
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
# 9. SAVE MODEL + TOKENIZER INTO v3
# ============================================================
trainer.save_model(STUDENT_MODEL_DIR)
tokenizer.save_pretrained(STUDENT_MODEL_DIR)

print("\n✅ Training complete. Saved to tinyllama-geocode-lora_v3.")
