import json
import matplotlib.pyplot as plt

# Load your JSON data
with open("/nas/mars/experiment_result/nsvqa/5_full_output/longvideobench_output.json", "r") as f:
    data = json.load(f)

ious, recalls, precisions = [], [], []
gt_durations = []

for item in data:
    pred_start, pred_end = item["nsvs"]["output"]
    positions = sorted(item["metadata"]["position"])

    if pred_start == 1 and pred_end == 1:
        pred_start = 0
        pred_end = item["metadata"]["frame_count"]

    # --- Ground Truth Segment Extraction Logic ---
    gt_segments = []
    if len(positions) == 2:
        gt_segments.append((positions[0], positions[1]))
    elif len(positions) == 3:
        gt_segments.append((positions[0], positions[1]))
        gt_segments.append((positions[2] - 200, positions[2] + 200))
    elif len(positions) == 4:
        gt_segments.append((positions[0], positions[1]))
        gt_segments.append((positions[2], positions[3]))
    else:
        continue  # skip if unexpected format

    # Merge all GT segments into one unioned range for IoU
    gt_all = []
    for seg_start, seg_end in gt_segments:
        gt_all.extend(range(seg_start, seg_end))
    gt_all = set(gt_all)

    pred_all = set(range(pred_start, pred_end))

    intersection = len(gt_all & pred_all)
    union = len(gt_all | pred_all)
    gt_len = len(gt_all)
    pred_len = len(pred_all)

    iou = intersection / union if union > 0 else 0
    recall = intersection / gt_len if gt_len > 0 else 0
    precision = intersection / pred_len if pred_len > 0 else 0

    ious.append(iou)
    recalls.append(recall)
    precisions.append(precision)
    gt_durations.append(gt_len)

# --- Plotting ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.hist(ious, bins=20, color="skyblue")
plt.title("IoU")

plt.subplot(1, 3, 2)
plt.hist(recalls, bins=20, color="lightgreen")
plt.title("Recall")

plt.subplot(1, 3, 3)
plt.hist(precisions, bins=20, color="salmon")
plt.title("Precision")

plt.tight_layout()
plt.savefig("iou_recall_precision.png")
plt.show()

# IoU vs Ground Truth Duration Plot
plt.figure(figsize=(10, 5))
plt.scatter(gt_durations, ious, alpha=0.7)
plt.xlabel("GT Segment Duration (frames)")
plt.ylabel("IoU")
plt.title("IoU vs Ground Truth Duration")
plt.grid(True)
plt.savefig("iou_vs_gt_duration.png")
plt.show()
