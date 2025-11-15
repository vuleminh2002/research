import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt = "Hello, this is a quick speed test."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

t0 = time.time()
out = model.generate(**inputs, max_new_tokens=512)
t1 = time.time()

print("Time:", t1 - t0)
print("Tokens/sec:", 512 / (t1 - t0))
