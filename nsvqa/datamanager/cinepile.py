from nsvqa.datamanager.manager import Manager

from tqdm import tqdm
import pandas as pd
import hashlib
import decord
import json
import copy
import os

class CinePile(Manager):
    def __init__(self):
        self._parquet_path = "/nas/mars/dataset/CinePile/test-00000-of-00001.parquet"
        self._datatset_path = "/nas/mars/dataset/CinePile/dataset/"

    def load_data(self):
        df = pd.read_parquet(self._parquet_path)

        temps = []
        for i in df.index:
            # marking where a "good question" is
            if ("before" in df.loc[i]["question"] or "after" in df.loc[i]["question"]) and df.loc[i]["question_category"] == "Setting and\nTechnical Analysis":
                temps.append(True)
            else:
                temps.append(False)

        # filtering out those good questions
        filtered_df = df[temps]
        
        data = []
        for idx, i in enumerate(filtered_df.index):
            entry = {}
            item = filtered_df.loc[i][:]
            
            entry["question"] = item["question"]
            entry["candidates"] = list(item["choices"])
            entry["correct_choice"] = int(item["answer_key_position"]) # this is basically a string that holds something in a list format
            entry["paths"] = {}
            entry["paths"]["video_path"] = os.path.join(self._datatset_path, item["videoID"] + ".mp4")
            
            entry["metadata"] = {}
            entry["metadata"]["video_id"] = item["videoID"]
            entry["metadata"]["movie_name"] = item["movie_name"] # name of movie it is from
            entry["metadata"]["year"] = int(item["year"]) # the year the movie is from
            entry["metadata"]["genre"] = list(item["genre"]) # the genre(s) of the movie
            entry["metadata"]["yt_clip_title"] = item["yt_clip_title"] # the name of the clip on youtube
            entry["metadata"]["yt_clip_link"] = item["yt_clip_link"] # link to the youtube clip
            entry["metadata"]["subtitles"] = item["subtitles"] # the subtitles for the clip but w/o timestamps
            entry["metadata"]["visual_reliance"] = item["visual_reliance"] # can you answer with just the subtitles
            entry["metadata"]["hard"] = item["hard_split"] # if this was a hard question
            entry["metadata"]["full_id"] = i # the position in the full, untouched dataset
            entry["metadata"]["filtered_id"] = idx # the position in the filtered dataset
            
            if os.path.exists(entry["paths"]["video_path"]):
                data.append(entry)
            
        return data
    
    def postprocess_data(self, nsvs_path):
        self._nsvs_path = nsvs_path
        run_name = self._nsvs_path.split('/')[-1].split('.')[0].replace('cinepile_', '')
        self._output_path_nsvqa = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/cinepile_nsvqa_{run_name}"
        self._output_path_full = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/cinepile_full_{run_name}"

        lvb_data = pd.read_parquet(self._parquet_path).to_dict(orient="records")
        with open(self._nsvs_path, "r") as f:
            nsvs_data = json.load(f)

        output_nsvqa = []    # nsvqa cropped video
        output_full = []     # entire video
        for entry_nsvs in tqdm(nsvs_data):
            found = False
            for entry in lvb_data:
                if entry["question"] == entry_nsvs["question"] and entry["videoID"] == entry_nsvs["metadata"]["video_id"]:
                    found = True

                    entry_full = copy.deepcopy(entry)

                    code = entry["question"] + entry["videoID"]
                    id = hashlib.sha256(code.encode()).hexdigest()
                    entry["videoID"] = id
                    video_file_name = id + ".mp4"

                    video_file_path = os.path.join(self._output_path_nsvqa, "cinepile", video_file_name)
                    self.crop_video(
                        entry_nsvs, 
                        save_path=video_file_path,
                        ground_truth=False
                    )

                    if os.path.exists(video_file_path): # if crop is successful
                        decordable = True
                        try:
                            _ = decord.VideoReader(video_file_path)
                        except:
                            decordable = False
                        if decordable:
                            output_nsvqa.append(entry)
                            output_full.append(entry_full)

            if found == False:
                print(f"Entry not found for question: {entry_nsvs['question']}")

        pd.DataFrame(output_nsvqa).to_parquet(os.path.join(self._output_path_nsvqa, "test-00000-of-00001.parquet"))
        pd.DataFrame(output_full).to_parquet(os.path.join(self._output_path_full, "test-00000-of-00001.parquet"))


# 0   to 42
# 43  to 85
# 86  to 128
# 129 to 171
# 172 to 214
