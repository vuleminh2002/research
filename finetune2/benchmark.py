# benchmark_raw.py
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)

prompt = "Test:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# benchmark 128 tokens
num = 128
start = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=num,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
end = time.time()

print("Time:", end - start)
print("Tokens/sec:", num / (end - start))
