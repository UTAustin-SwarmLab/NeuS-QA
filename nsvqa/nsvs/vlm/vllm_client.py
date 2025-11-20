from openai import OpenAI
import numpy as np
import base64
import math
import cv2

from nsvqa.utils.sigmoid import calibrate_sigmoid 
from nsvqa.nsvs.vlm.obj import DetectedObject


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
        # Encode a uint8 numpy array (image) as a JPEG and then base64 encode it.
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def detect(
        self,
        seq_of_frames: list[np.ndarray],
        scene_description: str,
        threshold: float
    ) -> DetectedObject:

        if "subtitle" in scene_description:
            subtitle_scene_description = scene_description.replace("subtitle_", "").replace("_", " ")
            parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Does the video have the subtitle 'this is very interesting' present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
            prompt = rf"Does the video have the subtitle '{subtitle_scene_description}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"
        else:
            object_scene_description = scene_description.replace("_", " ")
            parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Is there a cat present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
            prompt = rf"Is there a '{object_scene_description}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"

        # Encode each frame.
        encoded_images = [self._encode_frame(frame) for frame in seq_of_frames]

        # Build the user message: a text prompt plus one image for each frame.
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

        # Create a chat completion request.
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

        # Retrieve the list of TopLogprob objects.
        top_logprobs_list = chat_response.choices[0].logprobs.content[0].top_logprobs

        # Build a mapping from token text (stripped) to its probability.
        token_prob_map = {}
        for top_logprob in top_logprobs_list:
            token_text = top_logprob.token.strip()
            token_prob_map[token_text] = np.exp(top_logprob.logprob)

        # Extract probabilities for "Yes" and "No"
        yes_prob = token_prob_map.get("Yes", 0.0)
        no_prob = token_prob_map.get("No", 0.0)

        # Compute the normalized probability for "Yes": p_yes / (p_yes + p_no)
        if yes_prob + no_prob > 0:
            confidence = yes_prob / (yes_prob + no_prob)
        else:
            raise ValueError("No probabilities for 'Yes' or 'No' found in the response.")

        # print(f"Is detected: {is_detected}")
        # print(f"Confidence: {confidence:.3f}")


        probability = calibrate_sigmoid(confidence=confidence, false_threshold=threshold)

        return DetectedObject(
            name=scene_description,
            is_detected=is_detected,
            confidence=round(confidence, 3),
            probability=round(probability, 3)
        )

