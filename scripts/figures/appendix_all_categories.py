import os
import json
from collections import defaultdict

# === CONFIGURATION ===
base_dir = "/nas/mars/experiment_result/nsvqa/7_all_models"
modes = ["nsvqa", "base"]
results = {}

model_name = input("Enter model name (llavaonevision, qwen): ").strip()
model_name = model_name + "_supp"
if model_name not in ["llavaonevision_supp", "qwen_supp"]:
    import sys
    sys.exit(0)

for mode in modes:
    target_dir = os.path.join(base_dir, mode)
    
    model_path = next(
        (os.path.join(target_dir, d) for d in os.listdir(target_dir)
         if os.path.isdir(os.path.join(target_dir, d)) and model_name in d),
        None
    )
    
    if not model_path:
        print(f"Warning: {model_name} not found in {mode}")
        continue

    jsonl_file = next((f for f in os.listdir(model_path) if f.endswith(".jsonl")), None)
    if not jsonl_file:
        print(f"Warning: No JSONL file in {model_path}")
        continue

    file_path = os.path.join(model_path, jsonl_file)

    counts = defaultdict(lambda: [0, 0])  # correct, total
    total_correct = 0
    total = 0
    all_categories = set()

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            cat = entry["doc"]["question_category"]
            gt = entry["lvb_acc"]["answer"]
            pred = entry["lvb_acc"]["parsed_pred"]

            all_categories.add(cat)
            counts[cat][1] += 1
            if gt == pred:
                counts[cat][0] += 1
                total_correct += 1
            total += 1

    accuracies = {}
    for cat in sorted(all_categories):
        correct, total_cat = counts[cat]
        accuracies[cat] = 100 * correct / total_cat if total_cat else 0
    accuracies["Overall"] = 100 * total_correct / total if total else 0

    results[mode.upper()] = accuracies

# === PRINT RESULTS ===
for mode in ["NSVQA", "BASE"]:
    print(f"\n{mode}")
    if mode not in results:
        print("  (No data)")
        continue
    for cat, acc in results[mode].items():
        print(f"{cat}: {acc:.2f}")

