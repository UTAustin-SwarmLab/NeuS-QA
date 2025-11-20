from nsvqa.datamanager.manager import Manager

import pandas as pd
import os

class NextQA(Manager):
    def __init__(self):
        self._videos_path = "/nas/mars/dataset/NExTVideo/"
        self._annotations_path = "/nas/mars/dataset/nextqa/"

        self._categories = ["TN", "TP"]
        # TN and TP are "temporal next" and "temporal previous", might let us see how good the before/after checker is...
        # TC is "temporal concurrent", might be an interesting experiment to see how it handles with that ("while X is doing, what is Y doing" style of questions)

    def find_paths(self):
        vid_to_path = {}

        for subdir in os.listdir(self._videos_path):
          for video_file in os.listdir(os.path.join(self._videos_path, subdir)):
            video_id = int(video_file.split(".")[0]) # the one thing that does match is the video_id from the annotations is the filename of the video
            vid_to_path[video_id] = os.path.join(self._videos_path, subdir, video_file)
        return vid_to_path

    def load_data(self):
        data = []

        dataset = pd.read_csv(os.path.join(self._annotations_path, "val.csv"))

        vid_to_path = self.find_paths()

        for item_index in dataset.index:
            item = dataset.loc[item_index][:]

            if item["type"] in self._categories:
              entry = {}

              entry["question"] = item["question"]
              entry["candidates"] = [item["a0"], item["a1"], item["a2"], item["a3"], item["a4"]]
              # if it is comma separated instead of list, use "entry["candidates"] = ",".join(item["a0"], item["a1"], item["a2"], item["a3"], item["a4"])"
              entry["correct_choice"] = int(item["answer"])

              entry["paths"] = {}
              entry["paths"]["video_path"] = vid_to_path[int(item["video"])]
              # no subtitles, and hence no burned either

              entry["metadata"] = {}
              entry["metadata"]["frame_count"] = item["frame_count"]
              # not sure if the duration for lvbench is in seconds or frames, so I just called this frame_count
              entry["metadata"]["video_id"] = item["video"]
              entry["metadata"]["height"] = item["height"]
              entry["metadata"]["width"] = item["width"]
              entry["metadata"]["qid"] = item["qid"]
              # qid is "question id" -- some videos have multiple questions, so this denotes which question it is
              # entry["metadata"]["id"] = item["id"]
              # I just did id as the dataframe index... hope that works
              entry["metadata"]["question_category"] = item["type"]

              # if we only use TP, TN, and (maybe) TC, the following code will have add temporal category to the metadata
              # this can be used to check if the before/after part is predicting correctly
              entry["metadata"]["temporal_category"] = item["type"][1]

              data.append(entry)
        return data

    def postprocess_data(self):
        pass
