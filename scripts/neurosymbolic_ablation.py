import json
import logging
import os
import pickle
import traceback
from collections import defaultdict

import cv2
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from nsvqa.nsvs.vlm.obj import DetectedObject
from nsvqa.nsvs.vlm.vllm_client import VLLMClient

# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME = "OpenGVLab/InternVL2_5-8B"
RUN_NUM = 3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(f"no_nsvs_processing_{RUN_NUM}.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.disable(logging.CRITICAL)

CALIBRATION_THRESHOLD = 0.349  # vllm threshold
THRESHOLD = 0.5  # detection threshold
STRIDE = 10  # slide stride
WINDOW = 20  # window length
DEVICE_ID = 0  # GPU device index


class AblationVLM(VLLMClient):
    def detect(self, seq_of_frames, scene_description, threshold):
        logger.debug(f"Detecting scene: '{scene_description}' in {len(seq_of_frames)} frames")
        parsing_rule = """You must only return a Yes or No, and not both, to any question asked. 
You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. 
For example, if the question is "Is the video capable of answering the question 'Is the chalk blue' with the sequence of images?", 
the answer must only be 'Yes' or 'No'."""
        prompt = f"Does the video contain '{scene_description}'\n" f"[PARSING RULE]: {parsing_rule}"
        encoded_images = [self._encode_frame(frame) for frame in seq_of_frames]
        user_content = [
            {
                "type": "text",
                "text": f"The following is the sequence of images",
            }
        ]
        for encoded in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        content = chat_response.choices[0].message.content
        is_detected = "yes" in content.lower()

        top_logprobs_list = chat_response.choices[0].logprobs.content[0].top_logprobs
        token_prob_map = {}
        for top_logprob in top_logprobs_list:
            token_text = top_logprob.token.strip()
            token_prob_map[token_text] = np.exp(top_logprob.logprob)

        yes_prob = token_prob_map.get("Yes", 0.0)
        no_prob = token_prob_map.get("No", 0.0)

        if yes_prob + no_prob > 0:
            confidence = yes_prob / (yes_prob + no_prob)
        else:
            raise ValueError("No probabilities for 'Yes' or 'No' found in the response.")

        probability = self.calibrate(confidence=confidence, false_threshold=threshold)

        return DetectedObject(
            name=scene_description,
            is_detected=is_detected,
            confidence=round(confidence, 3),
            probability=round(probability, 3),
        )


def sliding_window(entry):  # answers "which sequence of `WINDOW` frames can best answer the query"
    query = entry["query"]
    frames = entry["images"]

    logger.info(f"Processing sliding window for query: '{query}' with {len(frames)} frames")

    model = AblationVLM(
        api_base=f"http://localhost:800{RUN_NUM}/v1",
        model=MODEL_NAME,
    )

    best = {"prob": -1.0, "start": 1, "end": 1}

    t = 0
    windows = list(range(0, len(frames), STRIDE))
    with tqdm(windows, desc=f"Sliding window (stride={STRIDE}, window={WINDOW})") as pbar:
        for t in pbar:
            end_idx = min(t + WINDOW, len(frames))
            seq = frames[t:end_idx]

            detect = model.detect(seq, query, CALIBRATION_THRESHOLD)
            prob = detect.probability
            is_detected = detect.is_detected

            pbar.set_postfix(
                {"best_prob": f"{best['prob']:.3f}", "current_prob": f"{prob:.3f}", "detected": is_detected}
            )

            if prob > best["prob"] and is_detected:
                best.update({"prob": prob, "start": t, "end": end_idx})
                logger.debug(f"New best window found: frames {t}-{end_idx}, prob={prob:.3f}")

    logger.info(f"Best window: frames {best['start']}-{best['end']} with probability {best['prob']:.3f}")
    if best["prob"] != -1.0:
        entry["frames_of_interest"] = list(range(best["start"], best["end"] + 1))
    else:
        entry["frames_of_interest"] = []

def frame_wise(entry):
    query = entry["query"]
    frames = entry["images"]

    logger.info(f"Processing frame_wise for query: '{query}' with {len(frames)} frames")

    model = AblationVLM(
        api_base=f"http://localhost:800{RUN_NUM}/v1",
        model=MODEL_NAME,
    )

    valid = []

    t = 0
    windows = range(len(frames))
    with tqdm(windows, desc=f"Framewise") as pbar:
        for t in pbar:
            f = [frames[t]]

            detect = model.detect(f, query, CALIBRATION_THRESHOLD)
            prob = detect.probability
            is_detected = detect.is_detected

            pbar.set_postfix(
                {"current_prob": f"{prob:.3f}", "detected": is_detected}
            )

            if prob > THRESHOLD and is_detected:
                valid.append(t)
                logger.debug(f"Added frame {t}, prob={prob:.3f}")

    entry["frames_of_interest"] = valid


def readTLV():
    base_path = "/nas/dataset/tlv-dataset-v1"
    logger.info(f"Reading TLV dataset from {base_path}")

    data = []
    total_files = 0
    processed_files = 0

    # Count total files first for progress bar
    for dataset_dir in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        for format_dir in os.listdir(dataset_path):
            format_path = os.path.join(dataset_path, format_dir)
            if not os.path.isdir(format_path):
                continue
            for filename in os.listdir(format_path):
                if filename.endswith(".pkl"):
                    total_files += 1

    logger.info(f"Found {total_files} pickle files to process")

    with tqdm(total=total_files, desc="Loading dataset files") as pbar:
        for dataset_dir in os.listdir(base_path):
            dataset_path = os.path.join(base_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue

            for format_dir in os.listdir(dataset_path):
                format_path = os.path.join(dataset_path, format_dir)
                if not os.path.isdir(format_path):
                    continue

                for filename in os.listdir(format_path):
                    if not filename.endswith(".pkl"):
                        continue

                    file_path = os.path.join(format_path, filename)
                    try:
                        with open(file_path, "rb") as f:
                            raw = pickle.load(f)
                            entry = {
                                "propositions": raw["proposition"],
                                "specification": raw["ltl_formula"],
                                "query": formatter(raw["ltl_formula"]),
                                "ground_truth": [i for sub in raw["frames_of_interest"] for i in sub],
                                "images": raw["images_of_frames"],
                                "type": {"dataset": dataset_dir, "format": format_dir},
                                "number_of_frame": raw["number_of_frame"],
                            }
                            data.append(entry)
                            processed_files += 1
                            pbar.set_postfix({"loaded": processed_files})
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                    finally:
                        pbar.update(1)

    logger.info(f"Successfully loaded {len(data)} entries from {processed_files} files")
    return data


def formatter(spec):
    # Replace logical operators with natural language
    spec = spec.replace("&", " and ")
    spec = spec.replace("|", " or ")
    spec = spec.replace("U", " until ")
    spec = spec.replace("F", " eventually ")
    spec = spec.replace("G", " always ")
    spec = spec.replace("X", " next ")

    # Remove quotes and parentheses
    spec = spec.replace('"', "")
    spec = spec.replace("'", "")
    spec = spec.replace("(", "")
    spec = spec.replace(")", "")

    # Remove extra spaces
    while "  " in spec:
        spec = spec.replace("  ", " ")
    spec = spec.strip()

    return spec

def process_no_nsvs():
    data = readTLV()

    if not data:
        logger.error("No data loaded, exiting")
        return

    output = []
    total_processing_time = 0

    prefix = MODEL_NAME.split("/")[1].split("_")[0].lower()
    suffix = "sw" if RUN_NUM % 2 == 0 else "fw"
    folder_name = f"{prefix}_{suffix}"
    folder_name = os.path.join("/nas/mars/experiment_result/nsvs/no_nsvs", folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with tqdm(enumerate(data), total=len(data), desc="Processing entries") as pbar:
        for i, entry in pbar:
            logger.info(f"Processing entry {i+1}/{len(data)}: {entry.get('specification', 'Unknown')}")

            start_time = time.time()
            try:
                if RUN_NUM % 2 == 0:
                    sliding_window(entry)
                else:
                    frame_wise(entry)
                logger.info(f"Successfully processed entry {i+1}")
            except Exception as e:
                logger.error(f"Error processing entry {i+1}: {str(e)}")
                traceback.print_exc()
                entry["frames_of_interest"] = []
            end_time = time.time()

            processing_time = end_time - start_time
            total_processing_time += processing_time
            entry["processing_time_seconds"] = round(processing_time, 3)

            output.append(entry)

            # Update progress bar with current stats
            avg_time = total_processing_time / (i + 1)
            pbar.set_postfix(
                {
                    "current_time": f"{processing_time:.1f}s",
                    "avg_time": f"{avg_time:.1f}s",
                }
            )

            logger.info(
                f"Entry {i+1} completed: frames {entry['frames_of_interest']}, "
                f"{len(entry['images'])} total frames, {processing_time:.3f}s"
            )

            # Save individual result
            entry_copy = entry.copy()
            entry_copy.pop("images", None)
            with open(f"{folder_name}/longvideobench_output_{i}.json", "w") as f:
                json.dump(entry_copy, f, indent=4)

            # Remove images from main entry to save memory
            entry.pop("images", None)

    logger.info(
        f"Processing completed! Total time: {total_processing_time:.2f}s, "
        f"Average time per entry: {total_processing_time/len(data):.2f}s"
    )

def evaluate():
    import json, os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import matplotlib.colors as mcolors

    folder = "output"
    out_path = "qwen.png"

    TP = FP = FN = 0
    # group F1s by number of propositions
    by_props = defaultdict(list)

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(folder, fname), "r") as f:
            data = json.load(f)

        # predictions from frames_of_interest (assumes [start, end])
        start, end = data["frames_of_interest"]
        start, end = int(start), int(end)
        pred = set(range(min(start, end), max(start, end) + 1))
        gt   = set(map(int, data["ground_truth"]))

        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        TP += tp; FP += fp; FN += fn

        # per-file F1
        precision_f = tp / (tp + fp) if (tp + fp) else 0.0
        recall_f    = tp / (tp + fn) if (tp + fn) else 0.0
        f1_file     = 2 * precision_f * recall_f / (precision_f + recall_f) if (precision_f + recall_f) else 0.0

        # x-axis: number of propositions
        n_props = len(data.get("propositions", []))
        by_props[n_props].append(float(f1_file))

    # micro-averaged metrics
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("Overall metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    if not by_props:
        print("No valid data to plot.")
        return

    # Prepare boxplot data in sorted order of proposition counts
    prop_counts = sorted(by_props.keys())
    box_data = [by_props[k] for k in prop_counts]

    # Colors
    base = mcolors.to_rgb("#1f77b4")           # light blue family
    darker = tuple(max(0.0, c * 0.75) for c in base)

    # Plot
    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(
        box_data,
        positions=prop_counts,
        widths=0.6,
        patch_artist=True,   # needed to color the boxes
        showfliers=False,    # hide outliers; toggle if you want them
    )

    # Color boxes (light blue)
    for box in bp['boxes']:
        box.set_facecolor(base)
        box.set_alpha(0.35)
        box.set_edgecolor(darker)
        box.set_linewidth(1.5)

    # Whiskers/medians in darker blue for contrast
    for element in ['whiskers', 'caps', 'medians']:
        for artist in bp[element]:
            artist.set_color(darker)
            artist.set_linewidth(1.5)

    plt.xlabel("number of propositions")
    plt.ylabel("F1 score")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved box plot to {out_path}")



    
def main():
    process_no_nsvs()
    evaluate()

if __name__ == "__main__":
    main()
