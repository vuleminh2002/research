#!/usr/bin/env python3
"""
Geocode Model Testing Script with Full Output Visibility
Tests a fine-tuned model's ability to determine which candidates are inside a rectangle
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "tinyllama-geocode-lora_final"
TEST_FILE = "geocode_train_vary_test.jsonl"
RESULTS_FILE = "geocode_test_results.jsonl"
SUMMARY_FILE = "geocode_test_results_summary.json"
MAX_NEW_TOKENS = 1024  # Increased to ensure full reasoning is generated

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_inside_ids(text):
    """Extract inside_ids from model output"""
    match = re.search(r"inside_ids:\s*\[(.*?)\]", text, re.DOTALL)
    if match:
        ids_str = match.group(1)
        # Extract quoted strings
        ids = re.findall(r"'([^']*)'", ids_str)
        return ids
    return []

def calculate_metrics(predicted_ids, true_ids):
    """Calculate precision, recall, F1, and exact match"""
    pred_set = set(predicted_ids)
    true_set = set(true_ids)
    
    if len(pred_set) == 0:
        precision = 0.0
        recall = 0.0 if len(true_set) > 0 else 1.0
    else:
        true_positives = len(pred_set & true_set)
        precision = true_positives / len(pred_set)
        recall = true_positives / len(true_set) if len(true_set) > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    exact_match = pred_set == true_set
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match
    }

# ============================================================
# MAIN TESTING FUNCTION
# ============================================================

def main():
    print("=" * 60)
    print("🧪 GEOCODE MODEL TESTING WITH FULL OUTPUT VISIBILITY")
    print("=" * 60)
    
    # Load model
    print(f"\n🧠 Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print("✅ Model loaded")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Load test data
    print(f"\n📂 Loading test data from {TEST_FILE}...")
    test_examples = []
    with open(TEST_FILE, 'r') as f:
        for line in f:
            test_examples.append(json.loads(line))
    print(f"✅ Loaded {len(test_examples)} test examples")
    
    # Run tests
    print("\n" + "=" * 60)
    print("🔍 RUNNING TESTS")
    print("=" * 60)
    
    results = []
    all_metrics = []
    
    for idx, example in enumerate(test_examples, 1):
        print("\n" + "=" * 60)
        print(f"📝 TEST EXAMPLE {idx}/{len(test_examples)}")
        print("=" * 60)
        
        # Prepare input
        input_text = example['input']
        ground_truth = example['output']
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_ids = inputs['input_ids']
        
        print("\n⏳ Generating prediction...")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        
        # Decode outputs
        full_output_with_tokens = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_only_with_tokens = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False)
        generated_clean = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Get token IDs
        generated_token_ids = outputs[0][len(input_ids[0]):].tolist()
        
        # Display input
        print("\n" + "-" * 60)
        print("🟦 INPUT (first 200 chars):")
        print("-" * 60)
        print(input_text[:200] + "...")
        
        # Display RAW output with special tokens
        print("\n" + "-" * 60)
        print("🔍 RAW MODEL OUTPUT (WITH SPECIAL TOKENS):")
        print("-" * 60)
        print(generated_only_with_tokens)
        print("-" * 60)
        
        # Display token IDs
        print("\n" + "-" * 60)
        print("🔢 GENERATED TOKEN IDs (first 50):")
        print("-" * 60)
        print(generated_token_ids[:50])
        if len(generated_token_ids) > 50:
            print(f"... and {len(generated_token_ids) - 50} more tokens")
        print("-" * 60)
        
        # Check for EOS token
        eos_token_id = tokenizer.eos_token_id
        print("\n" + "-" * 60)
        print("🛑 EOS TOKEN CHECK:")
        print("-" * 60)
        if eos_token_id in generated_token_ids:
            eos_position = generated_token_ids.index(eos_token_id)
            print(f"✅ EOS token </s> (ID: {eos_token_id}) found at position {eos_position}/{len(generated_token_ids)}")
            print(f"   Generation stopped naturally")
        else:
            print(f"❌ EOS token </s> (ID: {eos_token_id}) NOT found")
            print(f"   Model likely hit max_new_tokens limit ({MAX_NEW_TOKENS})")
            print(f"   Consider increasing MAX_NEW_TOKENS")
        
        # Display clean output
        print("\n" + "-" * 60)
        print("📝 CLEAN MODEL OUTPUT (special tokens removed):")
        print("-" * 60)
        print(generated_clean)
        print("-" * 60)
        
        # Check for reasoning
        print("\n" + "-" * 60)
        print("🧠 REASONING CHECK:")
        print("-" * 60)
        if "Reasoning:" in generated_clean:
            print("✅ Model DID generate reasoning")
            # Extract reasoning section
            reasoning_section = generated_clean.split("inside_ids:")[0]
            print("\nReasoning preview (first 500 chars):")
            print(reasoning_section[:500] + "...")
        else:
            print("❌ Model did NOT generate reasoning")
            print("   This may indicate:")
            print("   - Model wasn't trained properly to generate reasoning")
            print("   - max_new_tokens is too low")
            print("   - Model learned to skip reasoning")
        
        # Extract predicted IDs
        predicted_ids = extract_inside_ids(generated_clean)
        true_ids = extract_inside_ids(ground_truth)
        
        # Display ground truth
        print("\n" + "-" * 60)
        print("🏷️  GROUND TRUTH:")
        print("-" * 60)
        print(ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth)
        
        # Calculate metrics
        metrics = calculate_metrics(predicted_ids, true_ids)
        all_metrics.append(metrics)
        
        # Display evaluation
        print("\n" + "-" * 60)
        print("📊 EVALUATION:")
        print("-" * 60)
        print(f"Predicted IDs: {predicted_ids}")
        print(f"True IDs:      {true_ids}")
        print(f"Precision:     {metrics['precision']*100:.2f}%")
        print(f"Recall:        {metrics['recall']*100:.2f}%")
        print(f"F1 Score:      {metrics['f1']*100:.2f}%")
        print(f"Exact Match:   {'✅ YES' if metrics['exact_match'] else '❌ NO'}")
        
        # Save result
        results.append({
            'example_id': idx,
            'input': input_text,
            'predicted_output': generated_clean,
            'ground_truth': ground_truth,
            'predicted_ids': predicted_ids,
            'true_ids': true_ids,
            'metrics': metrics,
            'has_reasoning': "Reasoning:" in generated_clean,
            'has_eos_token': eos_token_id in generated_token_ids,
            'num_tokens_generated': len(generated_token_ids)
        })
    
    # Calculate overall metrics
    print("\n" + "=" * 60)
    print("📊 OVERALL RESULTS")
    print("=" * 60)
    
    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
    exact_match_rate = sum(m['exact_match'] for m in all_metrics) / len(all_metrics)
    
    print(f"\nTotal Examples:    {len(test_examples)}")
    print(f"Average Precision: {avg_precision*100:.2f}%")
    print(f"Average Recall:    {avg_recall*100:.2f}%")
    print(f"Average F1 Score:  {avg_f1*100:.2f}%")
    print(f"Exact Match Rate:  {exact_match_rate*100:.2f}% ({sum(m['exact_match'] for m in all_metrics)}/{len(all_metrics)})")
    
    # Check reasoning generation
    num_with_reasoning = sum(1 for r in results if r['has_reasoning'])
    print(f"\nReasoning Generated: {num_with_reasoning}/{len(results)} examples ({num_with_reasoning/len(results)*100:.1f}%)")
    
    num_with_eos = sum(1 for r in results if r['has_eos_token'])
    print(f"EOS Token Found:     {num_with_eos}/{len(results)} examples ({num_with_eos/len(results)*100:.1f}%)")
    
    # Save results
    print(f"\n💾 Saving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"✅ Results saved to: {RESULTS_FILE}")
    
    summary = {
        'total_examples': len(test_examples),
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'exact_match_rate': exact_match_rate,
        'reasoning_generation_rate': num_with_reasoning / len(results),
        'eos_token_rate': num_with_eos / len(results),
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {SUMMARY_FILE}")
    
    print("\n" + "=" * 60)
    print("🎉 TESTING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()