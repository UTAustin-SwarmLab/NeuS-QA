import os
import json

base_dir = "/nas/mars/experiment_result/nsvqa/7_all_models"

# Prompt user to select the directory inside base_dir
subdir = input("Enter subdirectory name inside base_dir: ").strip()
target_dir = os.path.join(base_dir, subdir)

# Get all model directories and initialize dictionary
model_dirs = {}
for d in os.listdir(target_dir):
    full_path = os.path.join(target_dir, d)
    if os.path.isdir(full_path):
        model_name = d.split("__")[1] if "__" in d else d
        model_dirs[model_name] = {}

results = {}

# Iterate through each model directory
for model_name, _ in model_dirs.items():
    # Find actual subdirectory (match endswith model_name, handles __ or not)
    full_dir = next(
        (d for d in os.listdir(target_dir)
         if d.endswith(model_name) and os.path.isdir(os.path.join(target_dir, d))),
        None
    )
    if not full_dir:
        print(f"Warning: Directory for {model_name} not found.")
        continue

    model_path = os.path.join(target_dir, full_dir)

    # Find any JSONL file in that directory
    jsonl_file = next(
        (f for f in os.listdir(model_path) if f.endswith(".jsonl")),
        None
    )
    if not jsonl_file:
        print(f"Warning: JSONL file not found in {model_path}")
        continue

    file_path = os.path.join(model_path, jsonl_file)

    # Initialize counters
    counts = {"T3E": [0, 0], "E3E": [0, 0], "T3O": [0, 0], "O3O": [0, 0]}
    counts2 = {"60": [0, 0], "600": [0, 0], "3600": [0, 0]}
    total_correct = 0
    total = 0

    # Read and process the file
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            cat = entry["doc"]["question_category"]
            gt = entry["lvb_acc"]["answer"]
            pred = entry["lvb_acc"]["parsed_pred"]
            if cat not in counts:
                continue

            dur = str(entry["doc"]["duration_group"])
            counts[cat][1] += 1
            counts2[dur][1] += 1
            if gt == pred:
                counts[cat][0] += 1
                counts2[dur][0] += 1
                total_correct += 1
            total += 1

    # Compute accuracies
    t3e = 100 * counts["T3E"][0] / counts["T3E"][1] if counts["T3E"][1] else 0
    e3e = 100 * counts["E3E"][0] / counts["E3E"][1] if counts["E3E"][1] else 0
    t3o = 100 * counts["T3O"][0] / counts["T3O"][1] if counts["T3O"][1] else 0
    o3o = 100 * counts["O3O"][0] / counts["O3O"][1] if counts["O3O"][1] else 0
    s600 = 100 * (counts2["60"][0] + counts2["600"][0]) / (counts2["60"][1] + counts2["600"][1]) if (counts2["60"][1] + counts2["600"][1]) else 0
    s3600 = 100 * counts2["3600"][0] / counts2["3600"][1] if counts2["3600"][1] else 0
    accuracy = 100 * total_correct / total if total else 0

    # Print results
    print(f"\nModel: {model_name}")
    print(f"T3E: {t3e:.2f}")
    print(f"E3E: {e3e:.2f}")
    print(f"T3O: {t3o:.2f}")
    print(f"O3O: {o3o:.2f}")
    print(f"600: {s600:.2f}")
    print(f"3600: {s3600:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Store for comparison
    results[model_name] = {"600": s600, "3600": s3600}

# If comparing "nsvqa" or "position" to "base"
if "nsvqa" in subdir or "position" in subdir:
    base_dir_to_compare = os.path.join(base_dir, "base")
    base_results = {}

    for d in os.listdir(base_dir_to_compare):
        full_path = os.path.join(base_dir_to_compare, d)
        if os.path.isdir(full_path):
            model_name = d.split("__")[1] if "__" in d else d
            jsonl_file = next((f for f in os.listdir(full_path) if f.endswith(".jsonl")), None)
            if not jsonl_file:
                continue

            file_path = os.path.join(full_path, jsonl_file)
            counts2 = {"60": [0, 0], "600": [0, 0], "3600": [0, 0]}

            with open(file_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    gt = entry["lvb_acc"]["answer"]
                    pred = entry["lvb_acc"]["parsed_pred"]
                    dur = str(entry["doc"]["duration_group"])
                    if dur not in counts2:
                        continue
                    counts2[dur][1] += 1
                    if gt == pred:
                        counts2[dur][0] += 1

            s600 = 100 * (counts2["60"][0] + counts2["600"][0]) / (counts2["60"][1] + counts2["600"][1]) if (counts2["60"][1] + counts2["600"][1]) else 0
            s3600 = 100 * counts2["3600"][0] / counts2["3600"][1] if counts2["3600"][1] else 0
            base_results[model_name] = {"600": s600, "3600": s3600}

    print("\n=== Comparison with base ===")
    fallback_model = "gpt-4o-2024-08-06"

    for model in results:
        base_model = model if model in base_results else fallback_model
        if base_model not in base_results:
            print(f"{model}: No base comparison found.")
            continue

        diff_600 = results[model]["600"] - base_results[base_model]["600"]
        diff_3600 = results[model]["3600"] - base_results[base_model]["3600"]
        print(f"{model}: 600 = {results[model]['600']:.2f} ({diff_600:+.2f}), 3600 = {results[model]['3600']:.2f} ({diff_3600:+.2f})")

    print("\n=== Per-sample change analysis ===")
    for model in results:
        base_model = model if model in base_results else fallback_model

        base_dir_model = next((d for d in os.listdir(base_dir_to_compare)
                               if d.endswith(base_model)), None)
        new_dir_model = next((d for d in os.listdir(target_dir)
                              if d.endswith(model)), None)

        if not base_dir_model or not new_dir_model:
            print(f"Skipping {model} due to missing directories.")
            continue

        base_path = os.path.join(base_dir_to_compare, base_dir_model)
        new_path = os.path.join(target_dir, new_dir_model)

        base_file = next((f for f in os.listdir(base_path) if f.endswith(".jsonl")), None)
        new_file = next((f for f in os.listdir(new_path) if f.endswith(".jsonl")), None)

        if not base_file or not new_file:
            print(f"Skipping {model} due to missing JSONL files.")
            continue

        base_jsonl = os.path.join(base_path, base_file)
        new_jsonl = os.path.join(new_path, new_file)

        with open(base_jsonl, "r") as f_base, open(new_jsonl, "r") as f_new:
            base_lines = f_base.readlines()
            new_lines = f_new.readlines()

        improved = 0
        worsened = 0
        total = 0

        for base_line, new_line in zip(base_lines, new_lines):
            base_entry = json.loads(base_line)
            new_entry = json.loads(new_line)

            gt = base_entry["lvb_acc"]["answer"]
            base_pred = base_entry["lvb_acc"]["parsed_pred"]
            new_pred = new_entry["lvb_acc"]["parsed_pred"]

            base_correct = (base_pred == gt)
            new_correct = (new_pred == gt)

            if not base_correct and new_correct:
                improved += 1
            elif base_correct and not new_correct:
                worsened += 1

            total += 1

        print(f"{model}: {improved}/{total} improved ({100*improved/total:.3f}%), {worsened}/{total} worsened ({100*worsened/total:.3f}%)")

