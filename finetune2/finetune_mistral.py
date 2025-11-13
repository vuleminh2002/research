import os
import gc
import math
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 1. Config
# ============================================================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_FILE = "geocode_train_randomized.jsonl"
OUTPUT_DIR = "mistral-geocode-lora"
MAX_LEN = 3072  # Based on your length stats; well below 4096 window


# ============================================================
# 2. Load dataset
# ============================================================
dataset = load_dataset("json", data_files={"train": DATA_FILE})
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]


# ============================================================
# 3. Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    use_fast=True,
)

# Add END token and set pad token
tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
END_ID = tokenizer.convert_tokens_to_ids("<END>")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4. Load model with QLoRA for A100
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Best on A100
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Resize embeddings to account for new token
model.resize_token_embeddings(len(tokenizer))

# Try to enable FlashAttention2 (if available)
try:
    model.config.use_flash_attention_2 = True
    print("✅ FlashAttention2 flag enabled.")
except Exception as e:
    print(f"⚠️ Could not enable FlashAttention2 explicitly: {e}")


# ============================================================
# 5. Apply LoRA (Mistral-specific modules)
# ============================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# 6. Tokenization with label masking (only learn on Response)
# ============================================================
def tokenize(example: Dict[str, str]) -> Dict[str, Any]:
    instruction = example["instruction"].strip()
    inp = example["input"].strip()
    out = example["output"].strip()

    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n{out}\n<END>"
    )

    # Tokenize full prompt (no padding here; we pad in collator)
    tok = tokenizer(
        full_prompt,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=False,
    )

    input_ids = tok["input_ids"]

    # Tokenize response only
    response_text = f"{out}\n<END>"
    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
    )["input_ids"]

    # Find response start in full sequence
    def find_subseq(full, sub):
        L = len(sub)
        for i in range(len(full) - L + 1):
            if full[i:i+L] == sub:
                return i
        return -1

    start = find_subseq(input_ids, response_ids)
    if start == -1:
        # Fallback if something weird happens (e.g. heavy truncation)
        start = max(0, len(input_ids) - len(response_ids))

    # Build labels: -100 before response, real tokens on/after response
    labels = []
    for idx, tok_id in enumerate(input_ids):
        if idx < start:
            labels.append(-100)
        else:
            labels.append(tok_id)

    return {
        "input_ids": input_ids,
        "attention_mask": tok["attention_mask"],
        "labels": labels,
    }


print("🧩 Tokenizing train/val...")
train_tok = train_ds.map(tokenize, remove_columns=train_ds.column_names)
val_tok = val_ds.map(tokenize, remove_columns=val_ds.column_names)


# ============================================================
# 7. Custom collator: dynamic padding + preserve -100 labels
# ============================================================
@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and attention_mask for padding
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels_list = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # Pad input_ids and attention_mask with tokenizer.pad
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Now pad labels manually with -100
        max_len = batch["input_ids"].shape[1]
        labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)

        for i, lab in enumerate(labels_list):
            length = lab.size(0)
            if length > max_len:
                labels[i, :] = lab[:max_len]
            else:
                labels[i, :length] = lab

        batch["labels"] = labels
        return batch


data_collator = CausalLMCollator(tokenizer=tokenizer, pad_to_multiple_of=8)


# ============================================================
# 8. TrainingArguments (optimized for A100)
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    bf16=True,  # A100 sweet spot
    report_to="none",
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    prediction_loss_only=True,  # do NOT store logits → avoids eval OOM
)


# ============================================================
# 9. Trainer
# ============================================================
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False  # Required when using gradient checkpointing

def compute_metrics(_):
    # We don't need per-token metrics here; eval_loss is enough
    return {}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ============================================================
# 10. Training loop + logging
# ============================================================
log_file = "training_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eval_loss", "perplexity"])

    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\n🚀 Starting Epoch {epoch + 1}/{training_args.num_train_epochs}")
        trainer.train()

        metrics = trainer.evaluate()
        loss = metrics["eval_loss"]
        ppl = math.exp(loss) if loss < 20 else float("inf")
        print(f"📊 Eval Loss: {loss:.4f} | Perplexity: {ppl:.2f}")

        writer.writerow([epoch + 1, loss, ppl])
        f.flush()


# ============================================================
# 11. Save final model + tokenizer
# ============================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete.")
print(f"💾 Model + LoRA saved to: {OUTPUT_DIR}")
print(f"📈 Metrics logged to: {log_file}")
