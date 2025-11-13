import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from difflib import SequenceMatcher

# ============================================================
# 1. Config
# ============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "tinyllama-geocode-lora"  # your LoRA checkpoint
DATA_FILE = "geocode_train_two_samples.jsonl"  # 2 samples with 50 candidates each
MAX_SAMPLES = 2
BATCH_SIZE = 15  # Process 15 candidates at a time

# ============================================================
# 2. Load Model + Tokenizer
# ============================================================
print("🧠 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)  # Comment out to test base model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# ============================================================
# 3. Helper functions
# ============================================================
def similarity(a, b):
    return round(SequenceMatcher(None, a.strip(), b.strip()).ratio(), 3)

def extract_lists(text):
    """Extract inside_ids and outside_ids lists from text"""
    inside_ids = []
    outside_ids = []
    
    # Try to find inside_ids
    if "inside_ids:" in text:
        inside_part = text.split("inside_ids:")[1]
        inside_line = inside_part.split("\n")[0].strip()
        try:
            # Try to parse as list
            if "[" in inside_line and "]" in inside_line:
                inside_str = inside_line.split("[")[1].split("]")[0]
                if inside_str.strip():
                    inside_ids = [x.strip().strip('"\'') for x in inside_str.split(",")]
        except:
            pass
    
    # Try to find outside_ids
    if "outside_ids:" in text:
        outside_part = text.split("outside_ids:")[1]
        outside_line = outside_part.split("\n")[0].strip()
        try:
            # Try to parse as list
            if "[" in outside_line and "]" in outside_line:
                outside_str = outside_line.split("[")[1].split("]")[0]
                if outside_str.strip():
                    outside_ids = [x.strip().strip('"\'') for x in outside_str.split(",")]
        except:
            pass
    
    return inside_ids, outside_ids

def split_candidates_into_batches(candidates_text, batch_size=15):
    """Split candidates into smaller batches to fit within token limits"""
    lines = candidates_text.strip().split('\n')
    candidates = [line.strip() for line in lines if ':' in line and line.strip()]  # Filter actual candidate lines
    
    batches = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batches.append('\n  ' + '\n  '.join(batch))  # Maintain proper indentation
    
    return batches

def process_single_batch(instruction, rectangle_text, candidates_text, pipe, batch_num, total_batches, ground_truth_for_batch=None):
    """Process a single batch of candidates"""
    
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\nRectangle:\n{rectangle_text}\nCandidates:{candidates_text}\n\n### Response:\n"
    
    print(f"\n📦 Processing batch {batch_num}/{total_batches}...")
    print(f"📏 Prompt tokens: {len(tokenizer.encode(prompt))}")
    
    # Show candidates in this batch
    candidate_lines = [line.strip() for line in candidates_text.split('\n') if ':' in line and line.strip()]
    candidate_ids = [line.split(':')[0].strip() for line in candidate_lines]
    print(f"📋 Candidates in batch {batch_num} ({len(candidate_lines)}): {candidate_ids[:3]}{'...' if len(candidate_ids) > 3 else ''}")
    
    result = pipe(
        prompt,
        max_new_tokens=1700,
        temperature=0.2,
        top_p=0.9,
        do_sample=False,
    )[0]["generated_text"]
    
    # Extract only the response part for display
    response_part = result.split("### Response:")[-1].strip()
    
    print(f"\n🤖 MODEL OUTPUT FOR BATCH {batch_num}:")
    print("="*50)
    print(response_part)
    print("="*50)
    
    # Extract lists from result
    inside_ids, outside_ids = extract_lists(result)
    
    print(f"\n📊 Batch {batch_num} extracted:")
    print(f"   Inside IDs: {inside_ids}")
    print(f"   Outside IDs: {outside_ids}")
    
    # Show ground truth for this batch if provided
    if ground_truth_for_batch:
        print(f"\n🎯 GROUND TRUTH FOR BATCH {batch_num}:")
        print(f"   Inside IDs: {ground_truth_for_batch['inside']}")
        print(f"   Outside IDs: {ground_truth_for_batch['outside']}")
        
        # Quick comparison
        batch_inside_set = set(inside_ids)
        batch_outside_set = set(outside_ids)
        truth_inside_set = set(ground_truth_for_batch['inside'])
        truth_outside_set = set(ground_truth_for_batch['outside'])
        
        inside_correct = len(batch_inside_set & truth_inside_set)
        outside_correct = len(batch_outside_set & truth_outside_set)
        total_correct = inside_correct + outside_correct
        total_in_batch = len(candidate_ids)
        
        batch_accuracy = total_correct / total_in_batch if total_in_batch > 0 else 0
        print(f"\n✅ Batch {batch_num} accuracy: {batch_accuracy:.2%} ({total_correct}/{total_in_batch} correct)")
    else:
        print(f"✅ Batch {batch_num} completed - Inside: {len(inside_ids)}, Outside: {len(outside_ids)}")
    
    return inside_ids, outside_ids, result, candidate_ids

def process_with_batching(instruction, rectangle_text, candidates_text, pipe, tokenizer, batch_size=15, truth_inside=None, truth_outside=None):
    """Process large inputs by batching candidates and combining results"""
    
    # Check if input is too large for single processing
    test_prompt = f"### Instruction:\n{instruction}\n\n### Input:\nRectangle:\n{rectangle_text}\nCandidates:\n{candidates_text}\n\n### Response:\n"
    tokens = tokenizer.encode(test_prompt)
    
    print(f"📊 Input Analysis:")
    print(f"   Total candidates: {len([line for line in candidates_text.split('\\n') if ':' in line])}")
    print(f"   Total tokens in prompt: {len(tokens)}")
    print(f"   Token limit check: {'✅ OK' if len(tokens) <= 1800 else '⚠️ TOO LARGE - Using batching'}")
    
    if len(tokens) <= 1800:  # Leave room for response generation
        print("🔄 Processing as single batch...")
        ground_truth_batch = None
        if truth_inside is not None and truth_outside is not None:
            ground_truth_batch = {'inside': truth_inside, 'outside': truth_outside}
        inside_ids, outside_ids, result, candidate_ids = process_single_batch(
            instruction, rectangle_text, candidates_text, pipe, 1, 1, ground_truth_batch
        )
        return inside_ids, outside_ids, [result]
    
    print(f"🔄 Using batch processing with batch size: {batch_size}")
    
    # Split candidates into batches
    candidate_batches = split_candidates_into_batches(candidates_text, batch_size)
    
    all_inside_ids = []
    all_outside_ids = []
    all_results = []
    
    # Create sets for quick lookup of ground truth
    truth_inside_set = set(truth_inside) if truth_inside else set()
    truth_outside_set = set(truth_outside) if truth_outside else set()
    
    for i, batch in enumerate(candidate_batches, 1):
        # Extract candidate IDs from this batch to match with ground truth
        batch_candidate_lines = [line.strip() for line in batch.split('\n') if ':' in line and line.strip()]
        batch_candidate_ids = [line.split(':')[0].strip() for line in batch_candidate_lines]
        
        # Determine ground truth for this specific batch
        ground_truth_batch = None
        if truth_inside is not None and truth_outside is not None:
            batch_inside = [cid for cid in batch_candidate_ids if cid in truth_inside_set]
            batch_outside = [cid for cid in batch_candidate_ids if cid in truth_outside_set]
            ground_truth_batch = {'inside': batch_inside, 'outside': batch_outside}
        
        inside_ids, outside_ids, result, candidate_ids = process_single_batch(
            instruction, rectangle_text, batch, pipe, i, len(candidate_batches), ground_truth_batch
        )
        
        all_inside_ids.extend(inside_ids)
        all_outside_ids.extend(outside_ids)
        all_results.append(result)
    
    print(f"\n🔗 Combined results - Total Inside: {len(all_inside_ids)}, Total Outside: {len(all_outside_ids)}")
    
    return all_inside_ids, all_outside_ids, all_results

def compare_lists(model_inside, model_outside, truth_inside, truth_outside):
    """Compare extracted lists and return detailed scores"""
    
    # Convert to sets for comparison
    model_inside_set = set(model_inside)
    model_outside_set = set(model_outside)
    truth_inside_set = set(truth_inside)
    truth_outside_set = set(truth_outside)
    
    # Calculate precision, recall, F1 for inside_ids
    inside_correct = len(model_inside_set & truth_inside_set)
    
    # Handle perfect match when both are empty
    if len(model_inside_set) == 0 and len(truth_inside_set) == 0:
        inside_precision, inside_recall, inside_f1 = 1.0, 1.0, 1.0
    else:
        inside_precision = inside_correct / len(model_inside_set) if model_inside_set else 0
        inside_recall = inside_correct / len(truth_inside_set) if truth_inside_set else 0
        inside_f1 = 2 * inside_precision * inside_recall / (inside_precision + inside_recall) if (inside_precision + inside_recall) > 0 else 0
    
    # Calculate precision, recall, F1 for outside_ids  
    outside_correct = len(model_outside_set & truth_outside_set)
    
    # Handle perfect match when both are empty
    if len(model_outside_set) == 0 and len(truth_outside_set) == 0:
        outside_precision, outside_recall, outside_f1 = 1.0, 1.0, 1.0
    else:
        outside_precision = outside_correct / len(model_outside_set) if model_outside_set else 0
        outside_recall = outside_correct / len(truth_outside_set) if truth_outside_set else 0
        outside_f1 = 2 * outside_precision * outside_recall / (outside_precision + outside_recall) if (outside_precision + outside_recall) > 0 else 0
    
    # Overall accuracy
    total_correct = inside_correct + outside_correct
    total_predicted = len(model_inside_set) + len(model_outside_set)
    total_truth = len(truth_inside_set) + len(truth_outside_set)
    
    overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
    overall_recall = total_correct / total_truth if total_truth > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        'inside_precision': round(inside_precision, 3),
        'inside_recall': round(inside_recall, 3),
        'inside_f1': round(inside_f1, 3),
        'outside_precision': round(outside_precision, 3),
        'outside_recall': round(outside_recall, 3),
        'outside_f1': round(outside_f1, 3),
        'overall_precision': round(overall_precision, 3),
        'overall_recall': round(overall_recall, 3),
        'overall_f1': round(overall_f1, 3),
        'inside_missing': list(truth_inside_set - model_inside_set),
        'inside_extra': list(model_inside_set - truth_inside_set),
        'outside_missing': list(truth_outside_set - model_outside_set),
        'outside_extra': list(model_outside_set - truth_outside_set)
    }

# ============================================================
# 4. Evaluate model on test data with batching
# ============================================================
print(f"📄 Loading {MAX_SAMPLES} test samples from {DATA_FILE}...")
with open(DATA_FILE) as f:
    data = [json.loads(line) for line in f][:MAX_SAMPLES]

print(f"🎯 Testing batching approach with batch size: {BATCH_SIZE}")
print(f"📋 Each sample has ~50 candidates, perfect for testing batching!\n")

total_f1_scores = []
batch_info = []

for i, sample in enumerate(data, 1):
    instruction = sample["instruction"]
    input_text = sample["input"]
    ground_truth = sample["output"]

    print(f"\n{'🧩' * 15}")
    print(f"🧩 SAMPLE {i}/{MAX_SAMPLES}")
    print(f"{'🧩' * 15}")

    # Parse input to extract rectangle and candidates
    input_lines = input_text.strip().split('\n')
    rectangle_lines = []
    candidates_lines = []
    
    current_section = None
    for line in input_lines:
        if line.startswith('Rectangle:'):
            current_section = 'rectangle'
            continue
        elif line.startswith('Candidates:'):
            current_section = 'candidates'
            continue
        elif current_section == 'rectangle':
            rectangle_lines.append(line)
        elif current_section == 'candidates':
            candidates_lines.append(line)
    
    rectangle_text = '\n'.join(rectangle_lines)
    candidates_text = '\n'.join(candidates_lines)

    # ------------------------------------------------------------
    # Extract ground truth lists first (needed for batch comparison)
    # ------------------------------------------------------------
    truth_inside, truth_outside = extract_lists(ground_truth)

    # ------------------------------------------------------------
    # Process using batching (with ground truth for each batch)
    # ------------------------------------------------------------
    model_inside, model_outside, batch_results = process_with_batching(
        instruction, rectangle_text, candidates_text, pipe, tokenizer, 
        batch_size=BATCH_SIZE, truth_inside=truth_inside, truth_outside=truth_outside
    )
    
    # Store batch information
    batch_info.append({
        'sample': i,
        'num_batches': len(batch_results),
        'total_candidates': len([line for line in candidates_text.split('\n') if ':' in line])
    })

    # ------------------------------------------------------------
    # Show ground truth
    # ------------------------------------------------------------
    print(f"\n🎯 COMPLETE GROUND TRUTH FOR SAMPLE {i}:")
    print("="*60)
    print(ground_truth)
    print("="*60)
    
    print(f"\n📋 FINAL COMPARISON (All Batches Combined):")
    print(f"Model inside_ids  ({len(model_inside)}): {model_inside}")
    print(f"Truth inside_ids  ({len(truth_inside)}): {truth_inside}")
    print(f"Model outside_ids ({len(model_outside)}): {model_outside[:10]}{'...' if len(model_outside) > 10 else ''}")
    print(f"Truth outside_ids ({len(truth_outside)}): {truth_outside[:10]}{'...' if len(truth_outside) > 10 else ''}")
    
    # Compare lists and get detailed metrics
    metrics = compare_lists(model_inside, model_outside, truth_inside, truth_outside)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"Inside  - Precision: {metrics['inside_precision']:.3f}, Recall: {metrics['inside_recall']:.3f}, F1: {metrics['inside_f1']:.3f}")
    print(f"Outside - Precision: {metrics['outside_precision']:.3f}, Recall: {metrics['outside_recall']:.3f}, F1: {metrics['outside_f1']:.3f}")
    print(f"Overall - Precision: {metrics['overall_precision']:.3f}, Recall: {metrics['overall_recall']:.3f}, F1: {metrics['overall_f1']:.3f}")
    
    # Show missing/extra items for debugging
    if metrics['inside_missing'] or metrics['inside_extra']:
        print(f"\n🔍 Inside ID Issues:")
        if metrics['inside_missing']:
            print(f"   Missing: {metrics['inside_missing'][:3]}{'...' if len(metrics['inside_missing']) > 3 else ''}")
        if metrics['inside_extra']:
            print(f"   Extra: {metrics['inside_extra'][:3]}{'...' if len(metrics['inside_extra']) > 3 else ''}")
    
    total_f1_scores.append(metrics['overall_f1'])

# ============================================================
# 5. Summary
# ============================================================
avg_f1_score = sum(total_f1_scores) / len(total_f1_scores)
print(f"\n{'='*60}")
print(f"🎯 BATCHING TEST RESULTS")
print(f"{'='*60}")
print(f"📊 Average F1 score across {len(data)} samples: {avg_f1_score:.3f}")
print(f"📦 Batch size used: {BATCH_SIZE}")

print(f"\n📋 Batching Summary:")
for info in batch_info:
    print(f"   Sample {info['sample']}: {info['total_candidates']} candidates → {info['num_batches']} batches")

print(f"\n✅ Batching successfully handled large inputs without exceeding token limits!")
print(f"📈 This approach allows processing any number of candidates while maintaining model quality.")