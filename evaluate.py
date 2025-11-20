from nsvqa.target_identification.target_identification import *
from nsvqa.nsvs.model_checker.frame_validator import *
from nsvqa.datamanager.longvideobench import *
from nsvqa.nsvs.video.read_video import *
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
    print(entry["video_path"])
    reader = Mp4Reader(path=entry["video_path"], sample_rate=sample_rate)
    video_data = reader.read_video()
    if "metadata" not in entry:
        entry["metadata"] = {}
    entry["metadata"]["fps"] = video_data["video_info"]["fps"]
    entry["metadata"]["frame_count"] = video_data["video_info"]["frame_count"]

    try:
        output, indices = run_nsvs(
            video_data,
            entry["video_path"],
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
    # data = loader.load_data()
    data = [
        {
            "video_path": "/nas/mars/dataset/longvideobench/burn-subtitles/mH9LdC7IFH8.mp4",
            "question": "In the scene, a man wearing glasses and dressed in a black suit is speaking to the camera under the golden sky. After the subtitle 'of that still uh in tanks and will be,' what happens on the screen?",
            "candidates": [
                "The screen switches to four cameras",
                "The screen switches to two cameras",
                "The camera view is enlarged",
                "The screen switches to three cameras",
                "The camera view is reduced"
            ]
        }
    ]
    
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

def postprocess(nsvqa_dir, vqa_dir, vlm_config):
    # loader = LongVideoBench()
    # loader.postprocess_data(output_dir)
    vqa(nsvqa_dir, vqa_dir, vlm_config, eval=False)
    

def main():
    current_split = 1
    total_splits = 30
    vlm_config = (0, "OpenGVLab/InternVL3_5-14B")

    nsvqa_dir = f"/nas/mars/experiment_result/nsvqa/9_post_submission/nsvqa_output/nsvqa_output_{current_split}.json"
    vqa_dir = f"/nas/mars/experiment_result/nsvqa/9_post_submission/vqa_output/vqa_output_{current_split}.json"

    run_nsvqa(nsvqa_dir, current_split, total_splits, vlm_config)
    postprocess(nsvqa_dir, vqa_dir, vlm_config)

if __name__ == "__main__":
    main()

