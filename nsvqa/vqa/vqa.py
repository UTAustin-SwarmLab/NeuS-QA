from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import numpy as np
import base64
import json
import tqdm
import cv2


NUM_SAMPLES = 16
NUM_WORKERS = 4

class VLLMClient:
    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        model="OpenGVLab/InternVL3_5-14B",
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def _encode_frame(self, frame):
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def multiple_choice(self, frames_by_cam: dict, question: str, candidates: list[str]) -> str:
        user_content = []
        user_content.append(
            {
                "type": "text",
                "text": "The following is the sequence of images",
            }
        )
        frames = list(frames_by_cam.values())[0]
        encoded_images = [self._encode_frame(frame) for frame in frames]
        for encoded in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        parsing_rule = "You must only return the letter of the answer choice, and nothing else. Do not include any other symbols, information, text, or justification in your answer. For example, if the correct answer is 'a) ...', you must only return 'a'."
        prompt = f"{question}\n"
        for candidate in candidates:
            prompt += f"{candidate}\n"
        prompt += f"\n[PARSING RULE]: {parsing_rule}"
        user_content.append({"type": "text", "text": prompt})

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            max_tokens=1,
            temperature=0.0,
        )
        return chat_response.choices[0].message.content.lower().strip()

class VLLMClientMultiprocessing(VLLMClient):
    def __init__(
        self,
        model,
        api_base,
        max_workers=NUM_WORKERS,
    ):
        super().__init__(model, api_base)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def multiple_choice_batch(self, batch_args):
        futures = [
            self.executor.submit(self.multiple_choice, *args) for args in batch_args
        ]
        
        results = []
        for future in tqdm.tqdm(futures, desc="Processing batch"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing a task: {e}")
                results.append(None)
                
        return results

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count

def load_video_frames(video_path, num_frames):
    frame_count = get_video_frame_count(video_path)
    if frame_count < num_frames:
        frame_indices = np.arange(frame_count)
    else:
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    images = []
    cap = cv2.VideoCapture(video_path)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            images.append(frame_rgb)
    cap.release()
    return images

def run_experiment(data, vllm_client, output_path, eval):
    results = []
    batch_args_all_calls = []

    for entry in data:
        frames = load_video_frames(entry["paths"]["video_path"], num_frames=NUM_SAMPLES)
        if not frames:
            continue
        for i in range(len(entry["candidates"])):
            entry["candidates"][i] = f"{chr(97+i)}) {entry['candidates'][i]}"
        batch_args_all_calls.append(({"main": frames}, entry["question"], entry["candidates"]))

    predicted_answers_all_calls = vllm_client.multiple_choice_batch(batch_args_all_calls)
    total_correct = 0
    for i, entry in enumerate(data):
        predicted_answer = predicted_answers_all_calls[i]
        output_dict = {
            "video_path": entry["paths"]["cropped_path"],
            "question": entry["question"],
            "candidates": entry["candidates"],
            "predicted_answer": predicted_answer
        }
        if eval:
            correct_answer = chr(97+entry["correct_choice"])
            is_correct = 1 if predicted_answer == correct_answer else 0
            total_correct += is_correct
            output_dict["correct_answer"] = correct_answer
            output_dict["is_correct"] = is_correct

        if "question_category" in entry:
            output_dict["question_category"] = entry["question_category"]
        results.append(output_dict)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    if eval:
        accuracy = total_correct / len(data)
        print(f"Accuracy: {accuracy:.2%}")
    else:
        for entry in results:
            print(entry["predicted_answer"])

def vqa(dataset_path, output_path, vlm_config, eval=False):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    vllm_client = VLLMClientMultiprocessing(
        model=vlm_config[1],
        api_base=f"http://localhost:800{vlm_config[0]}/v1"
    )
    run_experiment(
        data=data,
        vllm_client=vllm_client,
        output_path=output_path,
        eval=eval
    )

