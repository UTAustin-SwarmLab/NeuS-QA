import os
import json
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
base_dir = "/nas/mars/experiment_result/nsvqa/7_all_models"
subfolders = {"base": "Base", "position": "Ground Truth", "nsvqa": "NSVQA"}
model_folders = {
    "lmms-lab__llava-onevision-qwen2-7b-ov": "LLaVA-OneVision",
    "Qwen__Qwen2.5-VL-7B-Instruct": "Qwen 2.5",
    "gpt-4o-2024-08-06": "GPT-4o"
}
model_colors = {
    "LLaVA-OneVision": {"NSVQA": "#a3c6f4", "Ground Truth": "#6891cc"},
    "GPT-4o":           {"NSVQA": "#f4a3a3", "Ground Truth": "#d87b7b"},
    "Qwen 2.5":         {"NSVQA": "#b7e4c7", "Ground Truth": "#65b48e"}
}
bar_width = 0.15
bins = 5

# === FIND MAX DURATION ===
def find_max_duration(base_dir, subfolders, model_folders):
    max_dur = 0
    for sub in subfolders:
        subdir = os.path.join(base_dir, sub)
        for folder in os.listdir(subdir):
            if folder not in model_folders:
                continue
            fpath = os.path.join(subdir, folder)
            jsonl_file = next((f for f in os.listdir(fpath) if f.endswith(".jsonl")), None)
            if not jsonl_file:
                continue
            with open(os.path.join(fpath, jsonl_file)) as f:
                for line in f:
                    obj = json.loads(line)
                    dur = obj["doc"]["duration"]
                    if dur > max_dur:
                        max_dur = dur
    return max_dur

max_duration = find_max_duration(base_dir, subfolders, model_folders)

# === BINNING ===
bin_edges = np.linspace(0, max_duration, bins + 1)
bin_labels = [f"{int(a//60)}–{int(b//60)} min" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
x_a = np.arange(len(bin_labels)) * 1.3  # more space for wider figure (A)
x_b = np.arange(len(bin_labels)) * 0.8  # tighter spacing for deltas (B)

# === EXTRACT ACCURACIES ===
def extract_accuracies(folder_path):
    out = {}
    for folder in os.listdir(folder_path):
        if folder not in model_folders:
            continue
        model_name = model_folders[folder]
        fpath = os.path.join(folder_path, folder)
        jsonl_file = next((f for f in os.listdir(fpath) if f.endswith(".jsonl")), None)
        if not jsonl_file:
            continue

        counts = {i: [0, 0] for i in range(bins)}

        with open(os.path.join(fpath, jsonl_file)) as f:
            for line in f:
                obj = json.loads(line)
                duration = obj["doc"]["duration"]
                gt = obj["lvb_acc"]["answer"]
                pred = obj["lvb_acc"]["parsed_pred"]
                bin_idx = np.digitize(duration, bin_edges) - 1
                if bin_idx < 0 or bin_idx >= bins:
                    continue
                counts[bin_idx][1] += 1
                if gt == pred:
                    counts[bin_idx][0] += 1

        accs = {bin_labels[i]: 100 * counts[i][0] / counts[i][1] if counts[i][1] else 0 for i in range(bins)}
        out[model_name] = accs
    return out

# === LOAD ACCURACY DATA ===
all_results = {}
for subfolder, label in subfolders.items():
    folder_path = os.path.join(base_dir, subfolder)
    all_results[label] = extract_accuracies(folder_path)

# === REVISION A: Individual Bars (NSVQA + GT) ===
fig_a, ax_a = plt.subplots(figsize=(12, 6))
for i, model in enumerate(model_folders.values()):
    for label in ["NSVQA", "Ground Truth"]:
        offset = (i - 1) * (bar_width * 2.2) + (0 if label == "NSVQA" else bar_width)
        bar_x = x_a + offset
        vals = [all_results[label][model].get(lbl, 0) for lbl in bin_labels]
        color = model_colors[model][label]
        ax_a.bar(bar_x, vals, width=bar_width, label=f"{model} - {label}", color=color)

# Deduplicate legend
handles, labels = ax_a.get_legend_handles_labels()
seen = set()
unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
ax_a.legend(*zip(*unique), loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax_a.set_ylabel("Accuracy (%)")
ax_a.set_xlabel("Duration Range")
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(bin_labels)
ax_a.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("gt.pdf", dpi=300)
plt.close()

# === REVISION B: Delta Bars Only (NSVQA − GT) ===
fig_b, ax_b = plt.subplots(figsize=(8, 4))
for i, model in enumerate(model_folders.values()):
    offset = (i - 1) * (bar_width * 1.2)
    bar_x = x_b + offset
    nsvqa_vals = [all_results["NSVQA"][model].get(lbl, 0) for lbl in bin_labels]
    gt_vals = [all_results["Ground Truth"][model].get(lbl, 0) for lbl in bin_labels]
    deltas = [n - g for n, g in zip(nsvqa_vals, gt_vals)]
    plot_color = model_colors[model]["NSVQA"]
    ax_b.bar(bar_x, deltas, width=bar_width, label=f"{model}", color=plot_color)

# Deduplicate legend
handles, labels = ax_b.get_legend_handles_labels()
seen = set()
unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
ax_b.legend(*zip(*unique), loc="upper left", frameon=True, fontsize=10)
ax_b.axhline(0, color="black", linewidth=1.0)
ax_b.set_ylabel("Accuracy Gain Over Ground Truth (%)", fontsize=12)
ax_b.set_xlabel("Video Duration", fontsize=14)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(bin_labels, fontsize=12)
ax_b.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("gt_delta.pdf", dpi=300)
plt.close()
