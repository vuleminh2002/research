# finetune2

Code for LoRA fine-tuning TinyLlama on geocoding-style instruction data and utilities to analyze token lengths.

## Files
- `finetune_geocode_lora.py` — Fine-tune TinyLlama with PEFT-LoRA on `geocode_train_randomized.jsonl`.
- `number_token.py` — Estimates prompt/response token lengths for examples in `cc.jsonl`.
- `sft_data_generate.py` — Data generation utilities (if applicable).

## Quick start (Windows PowerShell)

1. Create/activate a Python 3.10+ environment (recommended) and install basics:

```powershell
python -m pip install --upgrade pip
python -m pip install transformers datasets peft accelerate bitsandbytes tqdm numpy
```

2. (Optional) For TinyLlama tokenizer compatibility, you may also need:

```powershell
python -m pip install protobuf sentencepiece tiktoken
```

3. Token length stats:

```powershell
python .\number_token.py
```

4. Fine-tune LoRA:

```powershell
python .\finetune_geocode_lora.py
```

Training outputs and large artifacts are ignored via `.gitignore` (e.g., `tinyllama-geocode-lora/`).

## Notes
- Ensure GPU + CUDA are available for faster training.
- If SSH push fails, set up your GitHub SSH keys or switch to HTTPS remote.
