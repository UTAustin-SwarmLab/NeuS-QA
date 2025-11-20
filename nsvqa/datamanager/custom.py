from nsvqa.datamanager.manager import Manager

from tqdm import tqdm
import json
import os


class Custom(Manager):
    def __init__(self, raw_data=None, postprocess_dir=None):
        self.raw_data = raw_data
        self.postprocess_dir = postprocess_dir

    def load_data(self) -> list:
        assert self.raw_data is not None

        ret = []
        for raw_entry in self.raw_data:
            ret.append({
                "paths": {
                    "video_path": raw_entry["video_path"]
                },
                "question": raw_entry["question"],
                "candidates": raw_entry["answer_choices"],
            })
        return ret

        
    def postprocess_data(self, nsvs_path): # use whenever you want to use vqa.py
        assert self.postprocess_dir is not None
        cropped_dir = os.path.join(os.path.dirname(self.postprocess_dir), "cropped_videos")
        os.makedirs(cropped_dir, exist_ok=True)

        with open(nsvs_path, "r") as f:
            nsvs_data = json.load(f)

        output = []
        for entry_nsvs in tqdm(nsvs_data):
            entry_nsvs["paths"]["cropped_path"] = os.path.join(cropped_dir, os.path.basename(entry_nsvs["paths"]["video_path"]))
            self.crop_video(
                entry_nsvs,
                save_path=entry_nsvs["paths"]["cropped_path"],
                ground_truth=False
            )
            if os.path.exists(entry_nsvs["paths"]["cropped_path"]): # if crop successful
                output.append(entry_nsvs)

        with open(self.postprocess_dir, "w") as f:
            json.dump(output, f, indent=4)
            
