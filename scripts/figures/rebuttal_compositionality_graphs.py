import json
from collections import defaultdict

rebuttal_file = "/nas/mars/experiment_result/nsvqa/5_full_output/longvideobench_rebuttal.json"
qwen_file = "/nas/mars/experiment_result/nsvqa/7_all_models/nsvqa/Qwen__Qwen2.5-VL-7B-Instruct/20250726_155411_samples_longvideobench_val_v.jsonl"

# Define the proposition groups. [3, 3] means 1-3, 4-6, >6. Empty list means individual propositions.
groups = [3, 3]

# Step 1: Create a dictionary from question to number of propositions
question_to_propositions = {}
with open(rebuttal_file, "r") as f:
    data = json.load(f)
    for entry in data:
        question = entry["question"]
        num_propositions = len(entry["puls"]["proposition"])
        if num_propositions > 0:  # Ignore 0 propositions
            question_to_propositions[question] = num_propositions

# Step 2 & 3: Iterate through the jsonl file, find propositions, and calculate accuracy
accuracy_by_proposition = defaultdict(lambda: {"correct": 0, "total": 0})

with open(qwen_file, "r") as f:
    for line in f:
        entry = json.loads(line)
        question = entry["doc"]["question"]
        
        if question in question_to_propositions:
            num_props = question_to_propositions[question]
            
            gt = entry["lvb_acc"]["answer"]
            pred = entry["lvb_acc"]["parsed_pred"]
            
            accuracy_by_proposition[num_props]["total"] += 1
            if gt == pred:
                accuracy_by_proposition[num_props]["correct"] += 1
        else:
            # This will print questions that are in qwen_file but not in the filtered rebuttal_data
            # print(f"Question not found in rebuttal data: {question}")
            pass

# Step 4: Calculate and print average accuracies based on groups
if not groups:
    print("Average accuracy per number of propositions (individual):")
    for num_props, data in sorted(accuracy_by_proposition.items()):
        accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"For {num_props} props, the average accuracy was {accuracy:.2f}% | {data['total']}")
else:
    accuracy_by_group = defaultdict(lambda: {"correct": 0, "total": 0})
    group_labels = []
    
    lower_bound = 1
    for group_size in groups:
        upper_bound = lower_bound + group_size - 1
        group_labels.append(f"{lower_bound}-{upper_bound}")
        lower_bound = upper_bound + 1
    group_labels.append(f">{upper_bound}")

    for num_props, data in accuracy_by_proposition.items():
        group_index = 0
        lower_bound = 1
        for i, group_size in enumerate(groups):
            upper_bound = lower_bound + group_size - 1
            if lower_bound <= num_props <= upper_bound:
                group_index = i
                break
            lower_bound = upper_bound + 1
        else:
            group_index = len(groups)
            
        group_label = group_labels[group_index]
        accuracy_by_group[group_label]["correct"] += data["correct"]
        accuracy_by_group[group_label]["total"] += data["total"]

    print("Average accuracy by proposition group:")
    for group_label in group_labels:
        data = accuracy_by_group[group_label]
        accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"For props {group_label}, the average accuracy was {accuracy:.2f}% | {data['total']}")
