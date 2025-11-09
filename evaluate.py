from nsvqa.target_identification.target_identification import *
from nsvqa.nsvs.model_checker.frame_validator import *
from nsvqa.datamanager.longvideobench import *
from nsvqa.nsvs.video.read_video import *
from nsvqa.datamanager.cinepile import *
from nsvqa.nsvs.vlm.obj import *
from nsvqa.nsvs.nsvs import *
from nsvqa.puls.puls import *

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

def exec_nsvs(entry, sample_rate, device, model_name): # Step 3
    print(entry["paths"]["video_path"])
    reader = Mp4Reader(path=entry["paths"]["video_path"], sample_rate=sample_rate)
    video_data = reader.read_video()
    entry["metadata"]["fps"] = video_data["video_info"]["fps"]
    entry["metadata"]["frame_count"] = video_data["video_info"]["frame_count"]

    try:
        output, indices = run_nsvs(
            video_data,
            entry["paths"]["video_path"],
            entry["puls"]["proposition"],
            entry["puls"]["specification"],
            device=device,
            model_name=model_name,
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

def run_nsvqa(output_dir):
    loader = LongVideoBench()
    # loader = CinePile()
    data = loader.load_data()
    
    output = []

    starting = 0
    ending = len(data)-1
    for i in range(starting, ending+1):
        print("\n" + "*"*50 + f" {i}/{len(data)-1} " + "*"*50)
        entry = data[i]
        exec_puls(entry)
        exec_target_identification(entry)
        exec_nsvs(entry, sample_rate=1, device=0, model_name="OpenGVLab/InternVL2_5-8B")
        exec_merge(entry)
        output.append(entry)
        # with open(f"junk/{i}.json", "w") as f:
        #     json.dump(output, f, indent=4)

    with open(output_dir, "w") as f:
        json.dump(output, f, indent=4)

def postprocess(output_dir):
    loader = LongVideoBench()
    loader.postprocess_data(output_dir)

def main():
    output_dir = "/nas/mars/experiment_result/nsvqa/5_full_output/longvideobench_rebuttal_1.json"
    run_nsvqa(output_dir)
    postprocess(output_dir)

if __name__ == "__main__":
    main()

