import json
import os
import subprocess
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def time_to_seconds(time_str):
    time_str = time_str.replace(" ", "")
    t = datetime.strptime(time_str, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def seconds_to_srt_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def burn_subtitles_on_video(video_path, subtitles_json_path, starting_timestamp, save_path,
                             font_size=24, font="Arial-Bold", color="white"):
    with open(subtitles_json_path, "r") as f:
        subtitles = json.load(f)

    bad = False
    prevstart = -1000
    minstartdelta = 100000
    for subtitle in subtitles:
        delta = time_to_seconds(subtitle["end"]) - time_to_seconds(subtitle["start"])
        minstartdelta = min(minstartdelta, abs(time_to_seconds(subtitle["start"]) - prevstart))
        prevstart = time_to_seconds(subtitle["start"])
        if abs(delta - 0.01) < 0.001 and subtitle["line"] != "[Music]":
            bad = True

    if bad:
        with open("err.txt", "a") as error_log:
            error_log.write(f"Bad subtitles detected for video: {video_path}\n")

    try:
        duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", video_path]
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        video_duration = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration for {video_path}: {e}")
        video_duration = float("inf")

    temp_srt_path = f"{save_path}_temp.srt"

    modified_subtitles = []
    for subtitle in subtitles:
        start = time_to_seconds(subtitle["start"]) + starting_timestamp
        end = (start + 1.5) if bad else (time_to_seconds(subtitle["end"]) + starting_timestamp)
        if start >= video_duration:
            continue
        end = min(end, video_duration)
        modified_subtitles.append({"start": start, "end": end, "line": subtitle["line"]})

    with open(temp_srt_path, "w", encoding="utf-8") as out:
        for i, entry in enumerate(modified_subtitles, start=1):
            start_str = seconds_to_srt_format(entry["start"])
            end_str = seconds_to_srt_format(entry["end"])
            out.write(f"{i}\n{start_str} --> {end_str}\n{entry['line']}\n\n")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"subtitles={temp_srt_path}:force_style='PrimaryColour=&HFFFFFF&,BackColour=&H000000&,BorderStyle=3'",
        "-c:a", "copy", "-y", save_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error for {video_path}: {e}")
        with open("err.txt", "a") as error_log:
            error_log.write(f"FFmpeg error for {video_path}: {e}\n")
    finally:
        try:
            os.remove(temp_srt_path)
        except FileNotFoundError:
            pass


def process_entry(entry):
    video_id = entry["video_id"]
    video_path = f"datasets/longvideobench/LongVideoBench/videos/{video_id}.mp4"
    subtitles_json_path = f"datasets/longvideobench/LongVideoBench/subtitles/{video_id}_en.json"
    save_path = f"datasets/longvideobench/burn-subtitles/{video_id}.mp4"

    # Skip if already processed
    if os.path.exists(save_path):
        return

    if not os.path.exists(video_path) or not os.path.exists(subtitles_json_path):
        with open("err.txt", "a") as error_log:
            if not os.path.exists(video_path):
                error_log.write(f"Video file not found: {video_path}\n")
            if not os.path.exists(subtitles_json_path):
                error_log.write(f"Subtitles file not found: {subtitles_json_path}\n")
        return

    try:
        starting_timestamp = entry["starting_timestamp_for_subtitles"]
    except KeyError as e:
        with open("err.txt", "a") as error_log:
            error_log.write(f"KeyError: {e} for video_id: {video_id}\n")
        return

    burn_subtitles_on_video(video_path, subtitles_json_path, starting_timestamp, save_path)


def main():
    with open("datasets/longvideobench/LongVideoBench/lvb_val.json", "r") as f:
        data = json.load(f)

    os.makedirs("datasets/longvideobench/burn-subtitles", exist_ok=True)

    num_workers = min(10, cpu_count())
    print(f"Using {num_workers} parallel workers.")
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_entry, data), total=len(data)))


if __name__ == "__main__":
    main()

