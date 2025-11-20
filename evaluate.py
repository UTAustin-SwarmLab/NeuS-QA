from nsvqa.target_identification.target_identification import *
from nsvqa.nsvs.model_checker.frame_validator import *
from nsvqa.datamanager.longvideobench import *
from nsvqa.nsvs.video.read_video import *
from nsvqa.datamanager.custom import *
from nsvqa.nsvs.vlm.obj import *
from nsvqa.nsvs.nsvs import *
from nsvqa.puls.puls import *
from nsvqa.vqa.vqa import *

import json
import os

def exec_puls(entry): # Step 1
    output = PULS(entry["question"])

    entry["puls"] = {}
    entry["puls"]["proposition"] = output["proposition"]
    entry["puls"]["specification"] = output["specification"]
    entry["puls"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_target_identification(entry): # Step 2
    output = identify_target(
        entry["question"],
        entry["candidates"],
        entry["puls"]["specification"],
        entry["puls"]["conversation_history"]
    )

    entry["target_identification"] = {}
    entry["target_identification"]["frame_window"] = output["frame_window"]
    entry["target_identification"]["explanation"] = output["explanation"]
    entry["target_identification"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_nsvs(entry, sample_rate, device, model): # Step 3
    print(entry["paths"]["video_path"])
    reader = Mp4Reader(path=entry["paths"]["video_path"], sample_rate=sample_rate)
    video_data = reader.read_video()
    if "metadata" not in entry:
        entry["metadata"] = {}
    entry["metadata"]["fps"] = video_data["video_info"]["fps"]
    entry["metadata"]["frame_count"] = video_data["video_info"]["frame_count"]

    try:
        output, indices = run_nsvs(
            video_data,
            entry["paths"]["video_path"],
            entry["puls"]["proposition"],
            entry["puls"]["specification"],
            device=device,
            model=model,
        )
    except Exception as e:
        entry["metadata"]["error"] = repr(e)
        output = [-1]
        indices = []
    
    entry["nsvs"] = {}
    entry["nsvs"]["output"] = output
    entry["nsvs"]["indices"] = indices

def exec_merge(entry): # Step 4
    inner = entry["target_identification"]["frame_window"].strip()[1:-1]
    parts = inner.split(',')
    result = []
    for part in parts:
        part = part.strip()
        match = re.search(r'([+-])\s*(\d+)', part)
        if match:
            sign, num = match.groups()
            result.append(int(sign + num))
        else:
            result.append(0)

    if entry["nsvs"]["output"] != [-1]:
        entry["frames_of_interest"] = [
            max(0,                                  int(entry["nsvs"]["output"][0] + result[0] * entry["metadata"]["fps"])),
            min(entry["metadata"]["frame_count"]-1, int(entry["nsvs"]["output"][1] + result[1] * entry["metadata"]["fps"]))
        ]
    else:
        entry["frames_of_interest"] = [-1]

def run_nsvqa(output_dir, current_split, total_splits, vlm_config):
    # loader = LongVideoBench()
    loader = Custom(
        raw_data=[
            {
                "video_path": "/nas/mars/dataset/longvideobench/burn-subtitles/mH9LdC7IFH8.mp4",
                "question": "What happens when wine shows up on the screen before the vineyards showed up on the screen?",
                "answer_choices": [
                    "A close up of the wine was shown",
                    "The wine was trashed",
                    "The wine was replaced with soda",
                    "The man in the blue shirt was talking"
                ]
            }
        ]
    )
    data = loader.load_data()
    
    output = []
    starting = (len(data) * (current_split-1)) // total_splits
    ending = (len(data) * current_split) // total_splits
    for i in range(starting, ending+1):
        print("\n" + "*"*50 + f" {i}/{len(data)-1} " + "*"*50)
        entry = data[i]
        exec_puls(entry)
        exec_target_identification(entry)
        exec_nsvs(entry, sample_rate=1, device=vlm_config[0], model=vlm_config[1])
        exec_merge(entry)
        output.append(entry)

    with open(output_dir, "w") as f:
        json.dump(output, f, indent=4)

def postprocess(nsvqa_dir, postprocess_dir):
    loader = Custom(postprocess_dir=postprocess_dir)
    loader.postprocess_data(nsvqa_dir)

def main():
    current_split = 1 # split between GPUs
    total_splits = 4
    vlm_config = (6, "OpenGVLab/InternVL3_5-14B") # device_number, model_name

    nsvqa_dir = f"/nas/mars/experiment_result/nsvqa/9_post_submission/nsvqa_output/nsvqa_output_{current_split}.json"
    vqa_dir = f"/nas/mars/experiment_result/nsvqa/9_post_submission/vqa_output/vqa_output_{current_split}.json"
    postprocess_dir = f"/nas/mars/experiment_result/nsvqa/9_post_submission/postprocess_output/postprocess_output_{current_split}.json"

    run_nsvqa(nsvqa_dir, current_split, total_splits, vlm_config)
    postprocess(nsvqa_dir, postprocess_dir)
    vqa(postprocess_dir, vqa_dir, vlm_config)

if __name__ == "__main__":
    main()

