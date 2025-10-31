import json
from collections import Counter

inside_counts = Counter()
with open("geocode_train_randomized.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        inside_counts[len(ex["output"].split("inside_ids: [")[1].split("]")[0].split(","))] += 1

print("Number of examples by count of inside_ids:", inside_counts)
